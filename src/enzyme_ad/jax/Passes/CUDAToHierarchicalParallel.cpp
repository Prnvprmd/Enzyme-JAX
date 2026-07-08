#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include <iostream>

namespace mlir {
  namespace enzyme {
#define GEN_PASS_DEF_CUDATOHIERARCHICALPARALLEL
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
  } // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

  // --- Bridge Pattern to convert GPU Launch into SCF Parallel Loops ---
  struct LaunchToParallelPattern : public OpRewritePattern<gpu::LaunchOp> {
    using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gpu::LaunchOp op, PatternRewriter &rewriter) const override {
      Location loc = op.getLoc();
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      SmallVector<Value> lowerBounds(6, zero);
      SmallVector<Value> steps(6, one);
      SmallVector<Value> upperBounds = {
	op.getGridSizeX(), op.getGridSizeY(), op.getGridSizeZ(),
	op.getBlockSizeX(), op.getBlockSizeY(), op.getBlockSizeZ()
      };

      auto parallelOp = rewriter.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
      Block *destBlock = parallelOp.getBody();
      Operation *yieldOp = destBlock->getTerminator();
      
      Block *sourceBlock = &op.getBody().front();
      
      for (auto &inst : llvm::make_early_inc_range(sourceBlock->without_terminator())) {
	rewriter.moveOpBefore(&inst, yieldOp);
      }

      SmallVector<Value> blockArgsReplacement = {
	parallelOp.getInductionVars()[0], parallelOp.getInductionVars()[1], parallelOp.getInductionVars()[2],
	parallelOp.getInductionVars()[3], parallelOp.getInductionVars()[4], parallelOp.getInductionVars()[5],
	upperBounds[0], upperBounds[1], upperBounds[2],
	upperBounds[3], upperBounds[4], upperBounds[5]
      };

      for (int i = 0; i < 12; ++i) {
	rewriter.replaceAllUsesWith(sourceBlock->getArgument(i), blockArgsReplacement[i]);
      }

      SmallVector<Operation*> toErase;
      destBlock->walk([&](Operation *innerOp) {
	if (auto blockId = dyn_cast<gpu::BlockIdOp>(innerOp)) {
	  int idx = (blockId.getDimension() == gpu::Dimension::x) ? 0 :
	    (blockId.getDimension() == gpu::Dimension::y) ? 1 : 2;
	  rewriter.replaceAllUsesWith(innerOp->getResult(0), parallelOp.getInductionVars()[idx]);
	  toErase.push_back(innerOp);
	} else if (auto threadId = dyn_cast<gpu::ThreadIdOp>(innerOp)) {
	  int idx = (threadId.getDimension() == gpu::Dimension::x) ? 3 :
	    (threadId.getDimension() == gpu::Dimension::y) ? 4 : 5;
	  rewriter.replaceAllUsesWith(innerOp->getResult(0), parallelOp.getInductionVars()[idx]);
	  toErase.push_back(innerOp);
	} else if (auto gridDim = dyn_cast<gpu::GridDimOp>(innerOp)) {
	  int idx = (gridDim.getDimension() == gpu::Dimension::x) ? 0 :
	    (gridDim.getDimension() == gpu::Dimension::y) ? 1 : 2;
	  rewriter.replaceAllUsesWith(innerOp->getResult(0), upperBounds[idx]);
	  toErase.push_back(innerOp);
	} else if (auto blockDim = dyn_cast<gpu::BlockDimOp>(innerOp)) {
	  int idx = (blockDim.getDimension() == gpu::Dimension::x) ? 3 :
	    (blockDim.getDimension() == gpu::Dimension::y) ? 4 : 5;
	  rewriter.replaceAllUsesWith(innerOp->getResult(0), upperBounds[idx]);
	  toErase.push_back(innerOp);
	}
      });

      for (auto opToErase : toErase) {
	rewriter.eraseOp(opToErase);
      }

      rewriter.eraseOp(op);
      return success();
    }
  };

  // --- Collapse Grid and Block loops into a 1D loop ---
  struct ParallelLoopCollapsePattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      unsigned numDims = op.getInductionVars().size();
      if (numDims <= 1) return failure();

      Location loc = op.getLoc();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      Value totalSize = one;
      SmallVector<Value> sizes;

      for (unsigned i = 0; i < numDims; ++i) {
        Value diff = rewriter.create<arith::SubIOp>(loc, op.getUpperBound()[i], op.getLowerBound()[i]);
        Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, op.getStep()[i], one);
        Value size = rewriter.create<arith::DivUIOp>(loc, rewriter.create<arith::AddIOp>(loc, diff, stepMinusOne), op.getStep()[i]);
        sizes.push_back(size);
        totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, size);
      }

      auto newLoop = rewriter.create<scf::ParallelOp>(loc, zero, totalSize, one, op.getInitVals());
      Block *destBlock = newLoop.getBody();
      Operation *yieldOp = destBlock->getTerminator();
      
      rewriter.setInsertionPoint(yieldOp);

      Value currentVal = newLoop.getInductionVars()[0];
      SmallVector<Value> decodedIvs(numDims);
      for (int i = numDims - 1; i >= 0; --i) {
        Value rem = rewriter.create<arith::RemUIOp>(loc, currentVal, sizes[i]);
        decodedIvs[i] = rewriter.create<arith::AddIOp>(loc, op.getLowerBound()[i], 
						       rewriter.create<arith::MulIOp>(loc, rem, op.getStep()[i]));
        if (i > 0) currentVal = rewriter.create<arith::DivUIOp>(loc, currentVal, sizes[i]);
      }

      Block *sourceBlock = op.getBody();
      for (auto &inst : llvm::make_early_inc_range(sourceBlock->without_terminator())) {
	rewriter.moveOpBefore(&inst, yieldOp);
      }

      for (unsigned i = 0; i < numDims; ++i) {
	rewriter.replaceAllUsesWith(sourceBlock->getArgument(i), decodedIvs[i]);
      }

      Operation *oldTerm = sourceBlock->getTerminator();
      if (isa<scf::ReduceOp>(oldTerm)) {
	rewriter.eraseOp(yieldOp);
	rewriter.moveOpBefore(oldTerm, destBlock, destBlock->end());
      }

      rewriter.replaceOp(op, newLoop.getResults());
      return success();
    }
  };

  struct ParallelLoopTilingPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    int targetMaxThreads;
    int targetBitWidth;
    int targetUnrollFactor;
    
    ParallelLoopTilingPattern(MLIRContext *context, int bitWidth, int unrollFactor, int maxThreads)
      : OpRewritePattern<scf::ParallelOp>(context),
        targetBitWidth(bitWidth), targetUnrollFactor(unrollFactor),
        targetMaxThreads(maxThreads) {}

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      if (op->hasAttr("tiled") || op->hasAttr("vectorized")) return failure();

      Location loc = op.getLoc();
      Value ub = op.getUpperBound()[0];
      Value lb = op.getLowerBound()[0];
      Value totalIters = rewriter.create<arith::SubIOp>(loc, ub, lb);
      Value numThreads = rewriter.create<arith::ConstantIndexOp>(loc, targetMaxThreads);

      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value threadChunk = rewriter.create<arith::DivUIOp>(loc, rewriter.create<arith::AddIOp>(loc, totalIters,  
											      rewriter.create<arith::SubIOp>(loc, numThreads, one)),numThreads);
      
      Type elementType = nullptr;
      for (auto &innerOp : *op.getBody()) {
        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) 
          elementType = cast<MemRefType>(load.getMemref().getType()).getElementType();
        if (elementType) break;
      }
      if (!elementType) elementType = rewriter.getF32Type();
    
      int64_t lanes = targetBitWidth / elementType.getIntOrFloatBitWidth();
      if (lanes <= 1) return failure();
      
      int64_t vectorFloorElements = lanes * targetUnrollFactor;
      Value vectorFloor = rewriter.create<arith::ConstantIndexOp>(loc, vectorFloorElements);
      Value tileSize = rewriter.create<arith::MaxSIOp>(loc, threadChunk, vectorFloor);
            
      auto tiledLoop = rewriter.create<scf::ParallelOp>(loc, op.getLowerBound(), op.getUpperBound(), ValueRange{tileSize}, op.getInitVals());
      rewriter.eraseOp(tiledLoop.getBody()->getTerminator());
      rewriter.setInsertionPointToStart(tiledLoop.getBody());
        
      Value iv = tiledLoop.getInductionVars()[0];
      Value upper = rewriter.create<arith::MinUIOp>(loc, rewriter.create<arith::AddIOp>(loc, iv, tileSize), op.getUpperBound()[0]); 
        
      auto innerLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{iv}, ValueRange{upper}, op.getStep(), tiledLoop.getRegionIterArgs());
      rewriter.eraseOp(innerLoop.getBody()->getTerminator());

      innerLoop->setAttr("tiled", rewriter.getUnitAttr());
      tiledLoop->setAttr("tiled", rewriter.getUnitAttr());

      rewriter.mergeBlocks(op.getBody(), innerLoop.getBody(), innerLoop.getInductionVars());
      rewriter.setInsertionPointToEnd(tiledLoop.getBody());
      rewriter.create<scf::ReduceOp>(loc, innerLoop.getResults());

      rewriter.replaceOp(op, tiledLoop.getResults());
      return success();
    }
  };

  // --- BULLETPROOF: Safe Barrier Handling ---
  struct BarrierFissionPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp parallelOp, PatternRewriter &rewriter) const override {
      gpu::BarrierOp barrier;
      for (auto &op : *parallelOp.getBody()) {
        if (auto b = dyn_cast<gpu::BarrierOp>(&op)) { barrier = b; break; }
      }
      if (!barrier) return failure();

      // CPU scf.parallel loops mapped to vector lanes execute in lock-step automatically. 
      // Splitting the loop naively breaks SSA uses because the bottom half still needs the 
      // registers loaded in the top half. The safest CPU transformation is to dissolve the barrier.
      rewriter.eraseOp(barrier);

      return success();
    }
  };

  struct LinearAccess {
    int64_t coeff = 0;    // coefficient of loop IV
    int64_t constant = 0; // invariant constant offset
    bool valid = true;
  };

  static LinearAccess analyzeLinearExpr(Value v, Value loopIV, scf::ForOp forOp) {
    // Base case: loop induction variable
    if (v == loopIV)
      return {1, 0, true};

    // Constant
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
      return {0, cst.value(), true};

    if (auto cst = v.getDefiningOp<arith::ConstantIntOp>())
      return {0, cst.value(), true};

    // Loop invariant value
    if (forOp.isDefinedOutsideOfLoop(v))
      return {0, 0, true};

    // Index cast
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
      return analyzeLinearExpr(cast.getIn(), loopIV, forOp);

    // Addition
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      auto lhs = analyzeLinearExpr(add.getLhs(), loopIV, forOp);
      auto rhs = analyzeLinearExpr(add.getRhs(), loopIV, forOp);

      if (!lhs.valid || !rhs.valid)
        return {0, 0, false};

      return {lhs.coeff + rhs.coeff, lhs.constant + rhs.constant, true};
    }

    // Subtraction
    if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      auto lhs = analyzeLinearExpr(sub.getLhs(), loopIV, forOp);
      auto rhs = analyzeLinearExpr(sub.getRhs(), loopIV, forOp);

      if (!lhs.valid || !rhs.valid)
        return {0, 0, false};

      return {lhs.coeff - rhs.coeff, lhs.constant - rhs.constant, true};
    }

    // Multiplication
    if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      auto lhs = analyzeLinearExpr(mul.getLhs(), loopIV, forOp);
      auto rhs = analyzeLinearExpr(mul.getRhs(), loopIV, forOp);

      if (!lhs.valid || !rhs.valid)
        return {0, 0, false};

      // Only allow one side to depend on IV
      if (lhs.coeff != 0 && rhs.coeff != 0)
        return {0, 0, false};

      // lhs = a*i + b, rhs = c
      if (rhs.coeff == 0)
        return {lhs.coeff * rhs.constant, lhs.constant * rhs.constant, true};

      // rhs = a*i + b, lhs = c
      if (lhs.coeff == 0)
        return {rhs.coeff * lhs.constant, rhs.constant * lhs.constant, true};
    }

    return {0, 0, false};
  }

  /*
    0	invariant → broadcast
    1	contiguous
    c > 1	strided/gather
    unknown	scalarize/reject
  */
  enum class AccessKind { Invariant, Contiguous, Strided, Unknown };

  template <typename T> static AccessKind classifyOpAccessKind(T load, scf::ForOp forOp, int64_t &strideOut) {
    Value iv = forOp.getInductionVar();
    Value idx = load.getIndices()[0];

    auto expr = analyzeLinearExpr(idx, iv, forOp);

    if (!expr.valid)
      return AccessKind::Unknown;

    strideOut = expr.coeff;

    if (expr.coeff == 0)
      return AccessKind::Invariant;

    if (expr.coeff == 1)
      return AccessKind::Contiguous;

    return AccessKind::Strided;
  }

  struct OpVectorizer {
    void loads(memref::LoadOp &load, PatternRewriter &rewriter, IRMapping &mapping, Location loc, Value mask, Value pad, VectorType vType, int vWidth, AccessKind kind = AccessKind::Contiguous, int64_t stride = 1) const;

    void stores(memref::StoreOp &store, PatternRewriter &rewriter, IRMapping &mapping, Location loc, Value mask) const;

    void arith_math_ops(Operation &Op, PatternRewriter &rewriter, IRMapping &mapping, Location loc, int vWidth) const;
  };


  SmallVector<Value> vectorizeRegion(mlir::Region &region, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder,
                                    Location nestedLoc, Type elemType, VectorType maskType, VectorType vType,
                                    SmallVector<Value> &partialSums, vector::CombiningKind redKind,
                                    llvm::DenseSet<Operation *> fusedMultiplies,
                                    llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth,
                                    Value finalIV);
  void vectorizeIfThenElse(scf::IfOp &ifOp, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder, Location nestedLoc,
                          Type elemType, VectorType maskType, VectorType vType, SmallVector<Value> &partialSums,
                          vector::CombiningKind redKind, llvm::DenseSet<Operation *> fusedMultiplies,
                          llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth,
                          Value finalIV);
  void vectorizeOp(Operation &innerOp, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder, Location nestedLoc,
                  Type elemType, VectorType maskType, VectorType vType, SmallVector<Value> &partialSums,
                  vector::CombiningKind redKind, llvm::DenseSet<Operation *> fusedMultiplies,
                  llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth, Value finalIV);

  SmallVector<Value> vectorizeRegion(mlir::Region &region, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder,
                                    Location nestedLoc, Type elemType, VectorType maskType, VectorType vType,
                                    SmallVector<Value> &partialSums, vector::CombiningKind redKind,
                                    llvm::DenseSet<Operation *> fusedMultiplies,
                                    llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth,
                                    Value finalIV) {

    if (region.empty())
      return {};

    for (Operation &IfOp : region.front().without_terminator()) {
      vectorizeOp(IfOp, mask, rewriter, mapping, nestedBuilder, nestedLoc, elemType, maskType, vType, partialSums, redKind,
                  fusedMultiplies, fmaCandidates, cstZero, vWidth, finalIV);
    }
    SmallVector<Value> result;
    if (auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator())) {
      for (Value v : yield.getOperands()) {
        result.push_back(mapping.lookupOrDefault(v));
      }
    }
    return result;
  }

  void vectorizeIfThenElse(scf::IfOp &ifOp, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder, Location nestedLoc,
                          Type elemType, VectorType maskType, VectorType vType, SmallVector<Value> &partialSums,
                          vector::CombiningKind redKind, llvm::DenseSet<Operation *> fusedMultiplies,
                          llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth,
                          Value finalIV) {
    Value cond = mapping.lookupOrDefault(ifOp.getCondition());
    if (!llvm::isa<VectorType>(cond.getType())) {
      cond = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, maskType, cond);
    }
    Value thenMask = nestedBuilder.create<arith::AndIOp>(nestedLoc, mask, cond);
    IRMapping thenMapping(mapping);
    IRMapping elseMapping(mapping);

    auto thenResult =
        vectorizeRegion(ifOp.getThenRegion(), thenMask, rewriter, thenMapping, nestedBuilder, nestedLoc, elemType, maskType, vType,
                        partialSums, redKind, fusedMultiplies, fmaCandidates, cstZero, vWidth, finalIV);

    Region &elseRegion = ifOp.getElseRegion();
    if (!elseRegion.empty()) {
      Value allTrue = nestedBuilder.create<arith::ConstantOp>(nestedLoc, DenseElementsAttr::get(maskType, true));
      Value notCond = nestedBuilder.create<arith::XOrIOp>(nestedLoc, cond, allTrue);
      Value elseMask = nestedBuilder.create<arith::AndIOp>(nestedLoc, mask, notCond);

      auto elseResult =
          vectorizeRegion(ifOp.getElseRegion(), elseMask, rewriter, elseMapping, nestedBuilder, nestedLoc, elemType, maskType,
                          vType, partialSums, redKind, fusedMultiplies, fmaCandidates, cstZero, vWidth, finalIV);

      SmallVector<Value> merged;
      for (auto [t, e] : llvm::zip(thenResult, elseResult)) {
        auto ensureVector = [&](Value v) {
          if (isa<VectorType>(v.getType()))
            return v;
          Value val = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, VectorType::get({vWidth}, v.getType()), v);
          return val;
        };
        t = ensureVector(t);
        e = ensureVector(e);
        merged.push_back(nestedBuilder.create<arith::SelectOp>(nestedLoc, cond, t, e));
      }
      for (auto [orig, newVal] : llvm::zip(ifOp.getResults(), merged)) {
        mapping.map(orig, newVal);
      }
      return;
    }
    for (auto [orig, newVal] : llvm::zip(ifOp.getResults(), thenResult)) {
      mapping.map(orig, newVal);
    }
  }



  void vectorizeOp(Operation &innerOp, Value mask, PatternRewriter &rewriter, IRMapping &mapping, OpBuilder &nestedBuilder, Location nestedLoc,
                  Type elemType, VectorType maskType, VectorType vType, SmallVector<Value> &partialSums,
                  vector::CombiningKind redKind, llvm::DenseSet<Operation *> fusedMultiplies,
                  llvm::DenseMap<Operation *, arith::MulFOp> fmaCandidates, Value cstZero, int vWidth, Value finalIV) {

    // Handle reductions
    if (innerOp.hasTrait<OpTrait::IsTerminator>()) {
      // llvm::outs() << "innerOp1: \n";
      if (auto origReduce = dyn_cast<scf::ReduceOp>(innerOp)) {
        // llvm::outs() << "innerOp2: \n";
        for (auto [i, operand] : llvm::enumerate(origReduce.getOperands())) {
          Value val = mapping.lookupOrDefault(operand);
          if (!llvm::isa<VectorType>(val.getType()))
            val = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, val);

          if (llvm::isa<FloatType>(elemType)) {
            partialSums[i] = (redKind == vector::CombiningKind::MUL)
                                ? nestedBuilder.create<arith::MulFOp>(nestedLoc, partialSums[i], val).getResult()
                                : nestedBuilder.create<arith::AddFOp>(nestedLoc, partialSums[i], val).getResult();
          } else {
            partialSums[i] = (redKind == vector::CombiningKind::MUL)
                                ? nestedBuilder.create<arith::MulIOp>(nestedLoc, partialSums[i], val).getResult()
                                : nestedBuilder.create<arith::AddIOp>(nestedLoc, partialSums[i], val).getResult();
          }
        }
      }
      return;
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(innerOp)) {
      vectorizeIfThenElse(ifOp, mask, rewriter, mapping, nestedBuilder, nestedLoc, elemType, maskType, vType, partialSums, redKind,
                          fusedMultiplies, fmaCandidates, cstZero, vWidth, finalIV);
      return;
    }

    // Skip fused multiplies
    if (fusedMultiplies.contains(&innerOp))
      return;

    // Generate FMAs
    if (auto addOp = dyn_cast<arith::AddFOp>(innerOp)) {
      if (fmaCandidates.count(addOp)) {
        arith::MulFOp mulOp = fmaCandidates[addOp];
        Value vL = mapping.lookupOrDefault(mulOp.getLhs());
        Value vR = mapping.lookupOrDefault(mulOp.getRhs());
        Value acc = (addOp.getLhs() == mulOp.getResult()) ? addOp.getRhs() : addOp.getLhs();
        Value vAcc = mapping.lookupOrDefault(acc);

        auto ensureVec = [&](Value v) {
          return llvm::isa<VectorType>(v.getType())
                    ? v
                    : nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, v).getResult();
        };

        Value fma = nestedBuilder.create<vector::FMAOp>(nestedLoc, ensureVec(vL), ensureVec(vR), ensureVec(vAcc));
        mapping.map(addOp.getResult(), fma);
        return;
      }
    }

    // Vectorise loads
    if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
      Value memref = mapping.lookupOrDefault(load.getMemref());
      if (!llvm::isa<MemRefType>(memref.getType()))
        return;

      SmallVector<Value> idxs;
      for (auto i : load.getIndices()) {
        Value idxVal = mapping.lookupOrDefault(i);
        if (llvm::isa<VectorType>(idxVal.getType())) {
          idxVal = nestedBuilder.create<vector::ExtractOp>(nestedLoc, idxVal, ArrayRef<int64_t>{0});
        }
        idxs.push_back(idxVal);
      }

      AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1,
                                                    nestedBuilder.getContext());

      auto readOp = nestedBuilder.create<vector::TransferReadOp>(nestedLoc, vType, memref, idxs, AffineMapAttr::get(map),
                                                                cstZero, mask, nestedBuilder.getBoolArrayAttr({false}));
      mapping.map(load.getResult(), readOp.getResult());
    }
    // Vectorise stores
    else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
      // llvm::outs() << "innerOp8: \n";
      Value memref = mapping.lookupOrDefault(store.getMemref());
      if (!llvm::isa<MemRefType>(memref.getType()))
        return;

      Value val = mapping.lookupOrDefault(store.getValueToStore());
      if (!llvm::isa<VectorType>(val.getType())) {
        VectorType correctVType = VectorType::get({vWidth}, val.getType());
        val = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, correctVType, val);
      }

      SmallVector<Value> idxs;
      for (auto i : store.getIndices()) {
        Value idxVal = mapping.lookupOrDefault(i);
        // THE FIX: Use the static vector::ExtractOp instead
        if (llvm::isa<VectorType>(idxVal.getType())) {
          idxVal = nestedBuilder.create<vector::ExtractOp>(nestedLoc, idxVal, ArrayRef<int64_t>{0});
        }
        idxs.push_back(idxVal);
      }

      AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1,
                                                    nestedBuilder.getContext());

      nestedBuilder.create<vector::TransferWriteOp>(nestedLoc, val, memref, idxs, AffineMapAttr::get(map), mask,
                                                    nestedBuilder.getBoolArrayAttr({false}));
    }
    // Vectorise Constants safely
    else if (auto constOp = dyn_cast<arith::ConstantOp>(innerOp)) {
      // llvm::outs() << "innerOp9: \n";
      Operation *scalarConst = nestedBuilder.clone(*constOp.getOperation());
      Type scalarType = scalarConst->getResult(0).getType();
      VectorType correctVType = VectorType::get({vWidth}, scalarType);
      Value broadcasted = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, correctVType, scalarConst->getResult(0));
      mapping.map(constOp.getResult(), broadcasted);
    }
    // General arith/math vectorisation
    else if (innerOp.getDialect()->getNamespace() == "arith" || innerOp.getDialect()->getNamespace() == "math") {
      SmallVector<Value> ops;
      for (auto o : innerOp.getOperands()) {
        Value v = mapping.lookupOrDefault(o);
        if (!llvm::isa<VectorType>(v.getType())) {
          VectorType correctVType = VectorType::get({vWidth}, v.getType());
          v = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, correctVType, v);
        }
        ops.push_back(v);
      }
      OperationState state(nestedLoc, innerOp.getName().getStringRef());
      state.addOperands(ops);
      for (Type t : innerOp.getResultTypes()) {
        state.addTypes(VectorType::get({vWidth}, t));
      }
      state.addAttributes(innerOp.getAttrs());
      Operation *vOp = nestedBuilder.create(state);
      for (unsigned i = 0; i < innerOp.getNumResults(); ++i)
        mapping.map(innerOp.getResult(i), vOp->getResult(i));
    }
  }

  void OpVectorizer::loads(memref::LoadOp &load, PatternRewriter &rewriter, IRMapping &mapping, Location loc, Value mask, Value pad, VectorType vType, int vWidth, AccessKind kind, int64_t stride) const {

    if (kind == AccessKind::Contiguous) {
      Value memref = mapping.lookupOrDefault(load.getMemref());
      if (!llvm::isa<MemRefType>(memref.getType()))
        return;

      SmallVector<Value> idxs;
      for (auto i : load.getIndices()) {
        Value idxVal = mapping.lookupOrDefault(i);
        if (llvm::isa<VectorType>(idxVal.getType())) {
          idxVal = rewriter.create<vector::ExtractOp>(loc, idxVal, ArrayRef<int64_t>{0});
        }
        idxs.push_back(idxVal);
      }

      AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1, rewriter.getContext());
      auto readOp = rewriter.create<vector::TransferReadOp>(loc, vType, memref, idxs, AffineMapAttr::get(map),
                                                          pad, mask, rewriter.getBoolArrayAttr({false}));

      mapping.map(load.getResult(), readOp.getResult());
    }
    else if (kind == AccessKind::Strided && stride > 1) {
      Value memref = mapping.lookupOrDefault(load.getMemref());
      if (!isa<MemRefType>(memref.getType()))
        return;

      Value baseIdx = mapping.lookupOrDefault(load.getIndices()[0]);

      // Gather expects scalar base.
      if (isa<VectorType>(baseIdx.getType()))
        baseIdx = rewriter.create<vector::ExtractOp>(loc, baseIdx, ArrayRef<int64_t>{0});

      // Construct vector of offsets: [0, stride, 2*stride, ...]
      SmallVector<int64_t> offsets;
      for (int64_t lane = 0; lane < vWidth; ++lane)
        offsets.push_back(lane * stride);

      auto idxVecTy = VectorType::get({vWidth}, rewriter.getIndexType());

      Value offsetVec = rewriter.create<arith::ConstantOp>(loc, DenseIntElementsAttr::get(idxVecTy, offsets));
      Value passThru = rewriter.create<arith::ConstantOp>(loc, vType, rewriter.getZeroAttr(vType));

      auto gather = rewriter.create<vector::GatherOp>(loc, vType, memref, baseIdx, offsetVec, mask, passThru);

      mapping.map(load.getResult(), gather.getResult());
    }
    else {
      Value memref = mapping.lookupOrDefault(load.getMemref());
      if (kind == AccessKind::Invariant && !isa<MemRefType>(memref.getType()))
        return;

      SmallVector<Value> idxs;
      for (auto idx : load.getIndices()) {
        Value mapped = mapping.lookupOrDefault(idx);

        // invariant loads should never use vector indices
        if (isa<VectorType>(mapped.getType()))
          mapped = rewriter.create<vector::ExtractOp>(loc, mapped, ArrayRef<int64_t>{0});

        idxs.push_back(mapped);
      }

      Value scalar = rewriter.create<memref::LoadOp>(loc, memref, idxs);
      Value vec = rewriter.create<vector::BroadcastOp>(loc, vType, scalar);
      mapping.map(load.getResult(), vec);
    }
  };

  void OpVectorizer::stores(memref::StoreOp &store, PatternRewriter &rewriter, IRMapping &mapping, Location loc, Value mask) const {

    Value val = mapping.lookupOrDefault(store.getValueToStore());
    Value memref = mapping.lookupOrDefault(store.getMemref());
    if (!llvm::isa<MemRefType>(memref.getType()))
      return;

    SmallVector<Value> idxs;
    for (Value idx : store.getIndices())
      idxs.push_back(mapping.lookupOrDefault(idx));

    AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1, rewriter.getContext());

    rewriter.create<vector::TransferWriteOp>(loc, val, store.getMemRef(), idxs, AffineMapAttr::get(map), mask,
                                                  rewriter.getBoolArrayAttr({false}));
  }

  void OpVectorizer::arith_math_ops(Operation &Op, PatternRewriter &rewriter, IRMapping &mapping, Location loc, int vWidth) const {
    SmallVector<Value> operands;
    for (Value operand : Op.getOperands()) {
      Value mapped = mapping.lookupOrDefault(operand);

      if (!isa<VectorType>(mapped.getType())) {
        mapped = rewriter.create<vector::BroadcastOp>(loc, VectorType::get({vWidth}, mapped.getType()), mapped);
      }
      operands.push_back(mapped);
    }

    OperationState state(loc, Op.getName().getStringRef());
    state.addOperands(operands);

    for (Type t : Op.getResultTypes())
      state.addTypes(VectorType::get({vWidth}, t));

    state.addAttributes(Op.getAttrs());
    Operation *vOp = rewriter.create(state);

    for (auto [oldRes, newRes] : llvm::zip(Op.getResults(), vOp->getResults()))
      mapping.map(oldRes, newRes);

  }


  struct VectorizeForLoops : public OpRewritePattern<scf::ParallelOp> {
    int targetBitWidth;

    OpVectorizer vectorizer;

    VectorizeForLoops(MLIRContext *ctx, int bitWidth) : OpRewritePattern<scf::ParallelOp>(ctx), targetBitWidth(bitWidth) {}

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {

      // Guard against re-processing
      if (op->hasAttr("loop-vectorized"))
        return failure();

      // Check for nested parallel loops
      bool containsParallel = false;
      for (Operation &innerOp : op.getRegion().getOps()) {
        innerOp.walk([&](scf::ParallelOp nested) {
          containsParallel = true;
          return WalkResult::interrupt();
        });
        if (containsParallel)
          break;
      }
      if (containsParallel)
        return failure();

      SmallVector<scf::ForOp> forLoops;

      op.walk([&](scf::ForOp forOp) {
        bool hasNestedLoop = false;

        forOp.walk([&](scf::ForOp nested) {
          if (nested != forOp) {
            hasNestedLoop = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (!hasNestedLoop)
          forLoops.push_back(forOp);
      });

      if (forLoops.empty())
        return failure();

      Location loc = op.getLoc();

      //----------------------------------------------------------------------
      // Determine element type
      //----------------------------------------------------------------------
      Type elemType = nullptr;
      op.walk([&](memref::LoadOp load) {
        if (!elemType)
          elemType = cast<MemRefType>(load.getMemref().getType()).getElementType();
      });

      if (!elemType)
        elemType = rewriter.getF32Type();

      unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
      int vWidth = targetBitWidth / (elemBitWidth ? elemBitWidth : 32);

      if (vWidth <= 1)
        return failure();

      VectorType vType = VectorType::get({vWidth}, elemType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      //----------------------------------------------------------------------
      // Clone whole parallel loop first
      //----------------------------------------------------------------------
      IRMapping mapping;
      auto newParallel = cast<scf::ParallelOp>(rewriter.clone(*op, mapping));

      //----------------------------------------------------------------------
      // Vectorize every inner scf.for
      //----------------------------------------------------------------------
      for (scf::ForOp oldFor : forLoops) {
        auto newFor = cast<scf::ForOp>(mapping.lookup(oldFor.getOperation()));

        //--------------------------------------------------------------------
        // Reject nested control-flow inside for (for now)
        //--------------------------------------------------------------------
        bool invalid = false;
        newFor.walk([&](Operation *nested) {
          if (nested == newFor.getOperation())
            return WalkResult::advance();

          if (isa<scf::ForOp>(nested) || isa<scf::IfOp>(nested) || isa<scf::ParallelOp>(nested)) {
            invalid = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (invalid)
          continue;

        rewriter.setInsertionPoint(newFor);

        //--------------------------------------------------------------------
        // Create widened step
        //--------------------------------------------------------------------
        Value oldStep = newFor.getStep();
        Value stepConst = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);

        Value newStep = rewriter.create<arith::MulIOp>(loc, oldStep, stepConst);

        //--------------------------------------------------------------------
        // Vectorize init args (loop-carried reductions)
        //--------------------------------------------------------------------
        SmallVector<Value> vecInitArgs;
        for (Value init : newFor.getInitArgs()) {
          Type t = init.getType();

          if (isa<IntegerType, FloatType>(t)) {
            auto vecTy = VectorType::get({vWidth}, t);
            Value broadcast = rewriter.create<vector::BroadcastOp>(loc, vecTy, init);
            vecInitArgs.push_back(broadcast);
          } else {
            vecInitArgs.push_back(init);
          }
        }

        //--------------------------------------------------------------------
        // Create vectorized loop
        //--------------------------------------------------------------------
        auto vecFor = rewriter.create<scf::ForOp>(loc, newFor.getLowerBound(), newFor.getUpperBound(), newStep, vecInitArgs);

        rewriter.setInsertionPointToStart(vecFor.getBody());

        //--------------------------------------------------------------------
        // Build vector induction variable
        //--------------------------------------------------------------------
        Value scalarIV = vecFor.getInductionVar();

        SmallVector<int64_t> laneValues;
        for (int i = 0; i < vWidth; ++i)
          laneValues.push_back(i);

        auto idxVecTy = VectorType::get({vWidth}, rewriter.getIndexType());

        auto laneAttr = DenseIntElementsAttr::get(idxVecTy, laneValues);
        Value laneOffsets = rewriter.create<arith::ConstantOp>(loc, laneAttr);
        Value baseIV = rewriter.create<vector::BroadcastOp>(loc, idxVecTy, scalarIV);
        Value vecIV = rewriter.create<arith::AddIOp>(loc, baseIV, laneOffsets);

        //--------------------------------------------------------------------
        // Local mapping for vectorized body
        //--------------------------------------------------------------------
        IRMapping bodyMap;
        bodyMap.map(newFor.getInductionVar(), vecIV);

        for (auto [oldArg, newArg] : llvm::zip(newFor.getRegionIterArgs(), vecFor.getRegionIterArgs())) {
          bodyMap.map(oldArg, newArg);
        }

        Value diff = rewriter.create<arith::SubIOp>(loc, newFor.getUpperBound(), scalarIV);
        Value clamped = rewriter.create<arith::MaxSIOp>(loc, diff, zeroIdx);
        Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, clamped);
        Value pad = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));

        //--------------------------------------------------------------------
        // Clone/vectorize body
        //--------------------------------------------------------------------
        for (Operation &bodyOp : newFor.getBody()->without_terminator()) {

          // Vectorize loads
          if (auto load = dyn_cast<memref::LoadOp>(bodyOp)) {
            int64_t stride = 0;
            auto kind = classifyOpAccessKind<memref::LoadOp>(load, newFor, stride);

            vectorizer.loads(load, rewriter, bodyMap, loc, mask, pad, vType, vWidth, kind, stride);
          }

          // Vectorize stores
          else if (auto store = dyn_cast<memref::StoreOp>(bodyOp)) {
            vectorizer.stores(store, rewriter, bodyMap, loc, mask);
          }

          // Vectorize arithmetic
          else if (bodyOp.getDialect()->getNamespace() == "arith" || bodyOp.getDialect()->getNamespace() == "math") {
            vectorizer.arith_math_ops(bodyOp, rewriter, bodyMap, loc, vWidth);
          }
        }

        //--------------------------------------------------------------------
        // Vectorized yield
        //--------------------------------------------------------------------
        auto oldYield = cast<scf::YieldOp>(newFor.getBody()->getTerminator());

        SmallVector<Value> yieldVals;
        for (Value val : oldYield.getOperands()) {
          Value mapped = bodyMap.lookupOrDefault(val);
          if (!isa<VectorType>(mapped.getType())) {
            mapped = rewriter.create<vector::BroadcastOp>(loc, VectorType::get({vWidth}, mapped.getType()), mapped);
          }
          yieldVals.push_back(mapped);
        }

        rewriter.create<scf::YieldOp>(loc, yieldVals);

        //--------------------------------------------------------------------
        // Horizontal reduction back to scalar
        //--------------------------------------------------------------------
        rewriter.setInsertionPointAfter(vecFor);

        SmallVector<Value> finalResults;
        for (Value vecRes : vecFor.getResults()) {
          auto vecTy = dyn_cast<VectorType>(vecRes.getType());
          if (!vecTy) {
            finalResults.push_back(vecRes);
            continue;
          }
          vector::CombiningKind redKind = getForReductionKind(oldFor);
          Value reduced = rewriter.create<vector::ReductionOp>(loc, redKind, vecRes);

          finalResults.push_back(reduced);
        }


        // Replace old scalar loop with reduced scalar results
        rewriter.replaceOp(newFor, finalResults);
      }

      newParallel->setAttr("loop-vectorized", rewriter.getUnitAttr());

      rewriter.replaceOp(op, newParallel.getResults());

      return success();
    }


    static vector::CombiningKind getForReductionKind(scf::ForOp forOp) {
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

      Value yielded = yield.getOperand(0);

      if (auto addf = yielded.getDefiningOp<arith::AddFOp>())
        return vector::CombiningKind::ADD;
      if (auto addi = yielded.getDefiningOp<arith::AddIOp>())
        return vector::CombiningKind::ADD;
      if (auto mulf = yielded.getDefiningOp<arith::MulFOp>())
        return vector::CombiningKind::MUL;
      if (auto muli = yielded.getDefiningOp<arith::MulIOp>())
        return vector::CombiningKind::MUL;

      llvm_unreachable("unsupported for-loop reduction kind");
    }
  };



  struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
    int targetBitWidth;
    int targetUnrollFactor;

    CUDAToVectorPattern(MLIRContext *ctx, int bitWidth, int unrollFactor)
      : OpRewritePattern<scf::ParallelOp>(ctx), 
        targetBitWidth(bitWidth), 
        targetUnrollFactor(unrollFactor) {}

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      //Guard against re-processing
      if (op->hasAttr("vectorized")) return failure();
      if (op.getLowerBound().size() != 1) return failure();

      Location loc = op.getLoc();

      // Check for nested parallel loops
      bool containsParallel = false;
      for (Operation &innerOp : op.getRegion().getOps()) {
        innerOp.walk([&](scf::ParallelOp nested) {
          containsParallel = true;
          return WalkResult::interrupt();
        });
        if (containsParallel) break;
      }
    if (containsParallel || checkParallelOp<scf::ForOp>(op))
      return failure();

      // Dynamic vector width calculation
      Type elemType = rewriter.getF32Type();
      Type elemType = nullptr;
      op.walk([&](memref::LoadOp load) {
      if (!elemType)
        elemType = cast<MemRefType>(load.getMemref().getType()).getElementType();
      });
      if (!elemType)
        elemType = rewriter.getF32Type();

      unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
      int vWidth = targetBitWidth / (elemBitWidth > 0 ? elemBitWidth : 32);
      VectorType vType = VectorType::get({vWidth}, elemType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
      Value stepConst = rewriter.create<arith::ConstantIndexOp>(loc, vWidth * targetUnrollFactor);

      auto getReductionKindFromOp = [](Operation *mathOp) -> std::optional<vector::CombiningKind> {
        if (isa<arith::AddFOp, arith::AddIOp>(mathOp)) return vector::CombiningKind::ADD;
        if (isa<arith::MulFOp, arith::MulIOp>(mathOp)) return vector::CombiningKind::MUL;
        if (isa<arith::MaximumFOp, arith::MaxSIOp>(mathOp)) return vector::CombiningKind::MAXSI;
        if (isa<arith::MinimumFOp, arith::MinSIOp>(mathOp)) return vector::CombiningKind::MINSI;
        return std::nullopt;
      };

      // Reduction kind detection
      vector::CombiningKind redKind = vector::CombiningKind::ADD;
      if (op.getNumReductions() > 0) {
        auto reduceOps = op.getBody()->getOps<scf::ReduceOp>();
        if (!reduceOps.empty()) {
          scf::ReduceOp firstReduce = *reduceOps.begin();
          if (firstReduce && firstReduce.getNumRegions() > 0) {
            Region &redRegion = firstReduce.getRegion(0);
            if (!redRegion.empty()) {
              auto it = redRegion.begin();
              if (it != redRegion.end()) {
                Block &redBlock = *it;
                for (Operation &mathOp : llvm::make_early_inc_range(redBlock.getOperations())) {
                  if (!mathOp.hasTrait<OpTrait::IsTerminator>()) {
                    if (auto kind = getReductionKindFromOp(&mathOp)) {
                      redKind = *kind;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }

      SmallVector<Value> vectorInits;
      for (auto [i, t] : llvm::enumerate(op.getResultTypes())) {
        Attribute identityAttr;
        if (redKind == vector::CombiningKind::MUL) {
          if (auto ft = llvm::dyn_cast<FloatType>(t))
            identityAttr = rewriter.getFloatAttr(ft, 1.0);
          else
            identityAttr = rewriter.getIntegerAttr(t, 1);
        } else if (redKind == vector::CombiningKind::MAXSI) {
          if (auto ft = llvm::dyn_cast<FloatType>(t))
            identityAttr = rewriter.getFloatAttr(ft, -std::numeric_limits<double>::infinity());
          else
            identityAttr = rewriter.getIntegerAttr(t, std::numeric_limits<int64_t>::min());
        } else {
          identityAttr = rewriter.getZeroAttr(t);
        }
        Value identity = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(identityAttr));
        vectorInits.push_back(rewriter.create<vector::BroadcastOp>(loc, VectorType::get({vWidth}, t), identity));
      }

      // First pass, check for FMAs
      llvm::DenseSet<Operation*> fusedMultiplies;
      llvm::DenseMap<Operation*, arith::MulFOp> fmaCandidates;

      for (Operation &innerOp : op.getBody()->getOperations()) {
        auto addOp = dyn_cast<arith::AddFOp>(innerOp);
        if (!addOp) continue;

        auto checkMul = [&](Value v) -> arith::MulFOp {
          Operation *defOp = v.getDefiningOp();
          if (!defOp || defOp == op || defOp->getName().getStringRef() != "arith.mulf") 
            return nullptr;
          if (!v.hasOneUse()) return nullptr; 
        
          auto m = cast<arith::MulFOp>(defOp);
          if (m->getBlock() == op.getBody()) return m;
          return nullptr;
        };

        if (auto m = checkMul(addOp.getLhs())) {
          fmaCandidates[addOp] = m;
          fusedMultiplies.insert(m);
        } else if (auto m = checkMul(addOp.getRhs())) {
          fmaCandidates[addOp] = m;
          fusedMultiplies.insert(m);
        }
      }

      // Second pass, transform atomics
      auto newLoop = rewriter.create<scf::ParallelOp>(loc, op.getLowerBound(), op.getUpperBound(), ValueRange{stepConst}, vectorInits,
						      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs, ValueRange /*iterArgs*/) {
        
							Value zeroIdx = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
							Value ub = op.getUpperBound()[0];
							Value cstZero = nestedBuilder.create<arith::ConstantOp>(nestedLoc, nestedBuilder.getZeroAttr(elemType));
        
							SmallVector<Value> partialSums = vectorInits;
        
							for (int u = 0; u < targetUnrollFactor; ++u) {
							  IRMapping mapping;

                SmallVector<int64_t> offsets;
                for (int i = 0; i < vWidth; ++i)
                  offsets.push_back(i);
                auto tmpVType = VectorType::get({vWidth}, nestedBuilder.getIndexType());
                auto laneConst =
                    nestedBuilder.create<arith::ConstantOp>(nestedLoc, DenseIntElementsAttr::get(tmpVType, offsets));
                Value offset = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, u * vWidth);
                Value iv = nestedBuilder.create<arith::AddIOp>(nestedLoc, ivs[0], offset);
                Value ivVec = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, tmpVType, iv);
                Value laneIV = nestedBuilder.create<arith::AddIOp>(nestedLoc, ivVec, laneConst);
                mapping.map(op.getInductionVars()[0], laneIV);

                Value diff = nestedBuilder.create<arith::SubIOp>(nestedLoc, ub, iv);
                Value clamped = nestedBuilder.create<arith::MaxSIOp>(nestedLoc, diff, zeroIdx);
                Value mask = nestedBuilder.create<vector::CreateMaskOp>(nestedLoc, maskType, clamped);
          
							  for (Operation &innerOp : op.getBody()->getOperations()) {
                  vectorizeOp(innerOp, mask, rewriter, mapping, nestedBuilder, nestedLoc, elemType, maskType, vType, partialSums,
                          redKind, fusedMultiplies, fmaCandidates, cstZero, vWidth, laneIV);
                }
							}
        
							// Deal with the loop terminator
							auto reduceOp = nestedBuilder.create<scf::ReduceOp>(nestedLoc, partialSums);
							for (unsigned i = 0; i < reduceOp.getNumRegions(); ++i) {
							  OpBuilder::InsertionGuard guard(nestedBuilder);
							  Region &region = reduceOp.getRegion(i);
          
							  if (region.empty()) {
							    nestedBuilder.createBlock(&region);
							  }
							  Block &redBlock = region.front();
          
							  if (redBlock.getNumArguments() == 0) {
							    redBlock.addArgument(vType, nestedLoc);
							    redBlock.addArgument(vType, nestedLoc);
							  }
          
							  nestedBuilder.setInsertionPointToEnd(&redBlock);
          
							  Value lhs = redBlock.getArgument(0);
							  Value rhs = redBlock.getArgument(1);
							  Value combined;
          
							  if (llvm::isa<FloatType>(elemType)) {
							    switch (redKind) {
							    case vector::CombiningKind::MUL: combined = nestedBuilder.create<arith::MulFOp>(nestedLoc, lhs, rhs).getResult(); break;
							    case vector::CombiningKind::MAXSI: combined = nestedBuilder.create<arith::MaximumFOp>(nestedLoc, lhs, rhs).getResult(); break;
							    case vector::CombiningKind::MINSI: combined = nestedBuilder.create<arith::MinimumFOp>(nestedLoc, lhs, rhs).getResult(); break;
							    default: combined = nestedBuilder.create<arith::AddFOp>(nestedLoc, lhs, rhs).getResult(); break;
							    }
							  } else {
							    switch (redKind) {
							    case vector::CombiningKind::MUL: combined = nestedBuilder.create<arith::MulIOp>(nestedLoc, lhs, rhs).getResult(); break;
							    case vector::CombiningKind::MAXSI: combined = nestedBuilder.create<arith::MaxSIOp>(nestedLoc, lhs, rhs).getResult(); break;
							    case vector::CombiningKind::MINSI: combined = nestedBuilder.create<arith::MinSIOp>(nestedLoc, lhs, rhs).getResult(); break;
							    default: combined = nestedBuilder.create<arith::AddIOp>(nestedLoc, lhs, rhs).getResult(); break;
							    }
							  }
							  nestedBuilder.create<scf::ReduceReturnOp>(nestedLoc, combined);
							}
						      });
      
      // Mark the loop as vectorised so it doesn't get reprocessed
      newLoop->setAttr("vectorized", rewriter.getUnitAttr());
      rewriter.setInsertionPointAfter(newLoop);

      // Do the final reduction
      if (newLoop.getNumResults() > 0) {
        SmallVector<Value> scalarResults;
        for (Value vRes : newLoop.getResults()) {
          scalarResults.push_back(rewriter.create<vector::ReductionOp>(loc, redKind, vRes));
        }
        rewriter.replaceOp(op, scalarResults);
      } else {
        rewriter.eraseOp(op);
      }
      
      return success();
    }

    template<typename Op>
    static bool checkParallelOp(scf::ParallelOp parallelOp) {
      bool containsOp = false;

      parallelOp.walk([&](Op nested) {
        containsOp = true;
        return WalkResult::interrupt();
      });

      return containsOp;
    }
  };


  struct CUDAToHierarchicalParallelPass : public enzyme::impl::CUDAToHierarchicalParallelBase<CUDAToHierarchicalParallelPass>  {
    using Base::Base;
  
    void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *ctx = &getContext();

      int finalWidth = (regBitWidth > 0) ? regBitWidth : 256; 
      int finalUnrollFactor = (unrollFactor > 0) ? unrollFactor : 4;
      // Added max threads fallback for your tiling pass
      int maxThreads = 256; 

      if (auto attr = module->getAttrOfType<StringAttr>("llvm.target_features")) {
        llvm::StringRef features = attr.getValue();
        if (features.contains("+avx512f")) finalWidth = 512;
        else if (features.contains("+avx2") || features.contains("+avx")) finalWidth = 256;
        else if (features.contains("+neon") || features.contains("+sse")) finalWidth = 128;
      }

      // Move gpu.allocs to memref.allocas
      SmallVector<gpu::AllocOp> allocsToMove;
      module.walk([&](gpu::AllocOp alloc) {
        if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(alloc.getType().getMemorySpace())) {
          if (intAttr.getInt() == 3) allocsToMove.push_back(alloc);
        }
      });

      for (auto alloc : allocsToMove) {
        func::FuncOp parentFunc = alloc->getParentOfType<func::FuncOp>();
        if (!parentFunc) continue; 
        OpBuilder hoistedBuilder(&parentFunc.getBody().front(), parentFunc.getBody().front().begin());
        auto stackMem = hoistedBuilder.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
        alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
        alloc.erase();
      }

      RewritePatternSet patterns(ctx);
      patterns.add<LaunchToParallelPattern>(ctx);
      patterns.add<ParallelLoopCollapsePattern>(ctx);
      patterns.add<BarrierFissionPattern>(ctx);
      patterns.add<ParallelLoopTilingPattern>(ctx, finalWidth, finalUnrollFactor, maxThreads);
      patterns.add<VectorizeForLoops>(ctx, finalWidth);
      patterns.add<CUDAToVectorPattern>(ctx, finalWidth, finalUnrollFactor);

      // This allows the LaunchToParallelPattern to generate scf.parallel loops,
      // and then immediately feeds those new loops into vectoriser, tiling, and fission patterns.
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
	signalPassFailure();
      }
    }
    
  };

} // namespace
