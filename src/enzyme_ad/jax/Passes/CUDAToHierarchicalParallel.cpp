#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

  LogicalResult matchAndRewrite(gpu::LaunchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> lowerBounds(6, zero);
    SmallVector<Value> steps(6, one);
    SmallVector<Value> upperBounds = {op.getGridSizeX(),  op.getGridSizeY(),
                                      op.getGridSizeZ(),  op.getBlockSizeX(),
                                      op.getBlockSizeY(), op.getBlockSizeZ()};

    auto parallelOp =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
    Block *destBlock = parallelOp.getBody();
    Operation *yieldOp = destBlock->getTerminator();

    Block *sourceBlock = &op.getBody().front();

    for (auto &inst :
         llvm::make_early_inc_range(sourceBlock->without_terminator())) {
      rewriter.moveOpBefore(&inst, yieldOp);
    }

    SmallVector<Value> blockArgsReplacement = {parallelOp.getInductionVars()[0],
                                               parallelOp.getInductionVars()[1],
                                               parallelOp.getInductionVars()[2],
                                               parallelOp.getInductionVars()[3],
                                               parallelOp.getInductionVars()[4],
                                               parallelOp.getInductionVars()[5],
                                               upperBounds[0],
                                               upperBounds[1],
                                               upperBounds[2],
                                               upperBounds[3],
                                               upperBounds[4],
                                               upperBounds[5]};

    for (int i = 0; i < 12; ++i) {
      rewriter.replaceAllUsesWith(sourceBlock->getArgument(i),
                                  blockArgsReplacement[i]);
    }

    SmallVector<Operation *> toErase;
    destBlock->walk([&](Operation *innerOp) {
      if (auto blockId = dyn_cast<gpu::BlockIdOp>(innerOp)) {
        int idx = (blockId.getDimension() == gpu::Dimension::x)   ? 0
                  : (blockId.getDimension() == gpu::Dimension::y) ? 1
                                                                  : 2;
        rewriter.replaceAllUsesWith(innerOp->getResult(0),
                                    parallelOp.getInductionVars()[idx]);
        toErase.push_back(innerOp);
      } else if (auto threadId = dyn_cast<gpu::ThreadIdOp>(innerOp)) {
        int idx = (threadId.getDimension() == gpu::Dimension::x)   ? 3
                  : (threadId.getDimension() == gpu::Dimension::y) ? 4
                                                                   : 5;
        rewriter.replaceAllUsesWith(innerOp->getResult(0),
                                    parallelOp.getInductionVars()[idx]);
        toErase.push_back(innerOp);
      } else if (auto gridDim = dyn_cast<gpu::GridDimOp>(innerOp)) {
        int idx = (gridDim.getDimension() == gpu::Dimension::x)   ? 0
                  : (gridDim.getDimension() == gpu::Dimension::y) ? 1
                                                                  : 2;
        rewriter.replaceAllUsesWith(innerOp->getResult(0), upperBounds[idx]);
        toErase.push_back(innerOp);
      } else if (auto blockDim = dyn_cast<gpu::BlockDimOp>(innerOp)) {
        int idx = (blockDim.getDimension() == gpu::Dimension::x)   ? 3
                  : (blockDim.getDimension() == gpu::Dimension::y) ? 4
                                                                   : 5;
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

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    unsigned numDims = op.getInductionVars().size();
    if (numDims <= 1)
      return failure();

    Location loc = op.getLoc();
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Value totalSize = one;
    SmallVector<Value> sizes;

    for (unsigned i = 0; i < numDims; ++i) {
      Value diff = rewriter.create<arith::SubIOp>(loc, op.getUpperBound()[i],
                                                  op.getLowerBound()[i]);
      Value stepMinusOne =
          rewriter.create<arith::SubIOp>(loc, op.getStep()[i], one);
      Value size = rewriter.create<arith::DivUIOp>(
          loc, rewriter.create<arith::AddIOp>(loc, diff, stepMinusOne),
          op.getStep()[i]);
      sizes.push_back(size);
      totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, size);
    }

    auto newLoop = rewriter.create<scf::ParallelOp>(loc, zero, totalSize, one,
                                                    op.getInitVals());
    Block *destBlock = newLoop.getBody();
    Operation *yieldOp = destBlock->getTerminator();

    rewriter.setInsertionPoint(yieldOp);

    Value currentVal = newLoop.getInductionVars()[0];
    SmallVector<Value> decodedIvs(numDims);
    for (int i = numDims - 1; i >= 0; --i) {
      Value rem = rewriter.create<arith::RemUIOp>(loc, currentVal, sizes[i]);
      decodedIvs[i] = rewriter.create<arith::AddIOp>(
          loc, op.getLowerBound()[i],
          rewriter.create<arith::MulIOp>(loc, rem, op.getStep()[i]));
      if (i > 0)
        currentVal = rewriter.create<arith::DivUIOp>(loc, currentVal, sizes[i]);
    }

    Block *sourceBlock = op.getBody();
    for (auto &inst :
         llvm::make_early_inc_range(sourceBlock->without_terminator())) {
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

  ParallelLoopTilingPattern(MLIRContext *context, int bitWidth,
                            int unrollFactor, int maxThreads)
      : OpRewritePattern<scf::ParallelOp>(context), targetBitWidth(bitWidth),
        targetUnrollFactor(unrollFactor), targetMaxThreads(maxThreads) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("tiled") || op->hasAttr("vectorized"))
      return failure();

    Location loc = op.getLoc();
    Value ub = op.getUpperBound()[0];
    Value lb = op.getLowerBound()[0];
    Value totalIters = rewriter.create<arith::SubIOp>(loc, ub, lb);
    Value numThreads =
        rewriter.create<arith::ConstantIndexOp>(loc, targetMaxThreads);

    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value threadChunk = rewriter.create<arith::DivUIOp>(
        loc,
        rewriter.create<arith::AddIOp>(
            loc, totalIters,
            rewriter.create<arith::SubIOp>(loc, numThreads, one)),
        numThreads);

    Type elementType = nullptr;
    for (auto &innerOp : *op.getBody()) {
      if (auto load = dyn_cast<memref::LoadOp>(innerOp))
        elementType =
            cast<MemRefType>(load.getMemref().getType()).getElementType();
      if (elementType)
        break;
    }
    if (!elementType)
      elementType = rewriter.getF32Type();

    int64_t lanes = targetBitWidth / elementType.getIntOrFloatBitWidth();
    if (lanes <= 1)
      return failure();

    int64_t vectorFloorElements = lanes * targetUnrollFactor;
    Value vectorFloor =
        rewriter.create<arith::ConstantIndexOp>(loc, vectorFloorElements);
    Value tileSize =
        rewriter.create<arith::MaxSIOp>(loc, threadChunk, vectorFloor);

    auto tiledLoop = rewriter.create<scf::ParallelOp>(
        loc, op.getLowerBound(), op.getUpperBound(), ValueRange{tileSize},
        op.getInitVals());
    rewriter.eraseOp(tiledLoop.getBody()->getTerminator());
    rewriter.setInsertionPointToStart(tiledLoop.getBody());

    Value iv = tiledLoop.getInductionVars()[0];
    Value upper = rewriter.create<arith::MinUIOp>(
        loc, rewriter.create<arith::AddIOp>(loc, iv, tileSize),
        op.getUpperBound()[0]);

    auto innerLoop = rewriter.create<scf::ParallelOp>(
        loc, ValueRange{iv}, ValueRange{upper}, op.getStep(),
        tiledLoop.getRegionIterArgs());
    rewriter.eraseOp(innerLoop.getBody()->getTerminator());

    innerLoop->setAttr("tiled", rewriter.getUnitAttr());
    tiledLoop->setAttr("tiled", rewriter.getUnitAttr());

    rewriter.mergeBlocks(op.getBody(), innerLoop.getBody(),
                         innerLoop.getInductionVars());
    rewriter.setInsertionPointToEnd(tiledLoop.getBody());
    rewriter.create<scf::ReduceOp>(loc, innerLoop.getResults());

    rewriter.replaceOp(op, tiledLoop.getResults());
    return success();
  }
};

// --- BULLETPROOF: Safe Barrier Handling ---
struct BarrierFissionPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    gpu::BarrierOp barrier;
    for (auto &op : *parallelOp.getBody()) {
      if (auto b = dyn_cast<gpu::BarrierOp>(&op)) {
        barrier = b;
        break;
      }
    }
    if (!barrier)
      return failure();

    // CPU scf.parallel loops mapped to vector lanes execute in lock-step
    // automatically. Splitting the loop naively breaks SSA uses because the
    // bottom half still needs the registers loaded in the top half. The safest
    // CPU transformation is to dissolve the barrier.
    rewriter.eraseOp(barrier);

    return success();
  }
};

struct AffineExprInfo {
  // True iff the expression is affine w.r.t. the chosen vector IV.
  bool valid = true;

  // Coefficient of the vector IV.
  int64_t coeff = 0;

  // Constant offset.
  //
  // Expression represented is:
  //
  //      coeff * IV + offset
  //
  int64_t offset = 0;
};

static inline AffineExprInfo invalidAffine() {
  return {.valid = false, .coeff = 0, .offset = 0};
}

static inline AffineExprInfo affineCoeff(int64_t coeff) {
  return {.valid = true, .coeff = coeff, .offset = 0};
}

static inline AffineExprInfo affineConstant(int64_t value) {
  return {.valid = true, .coeff = 0, .offset = value};
}

static inline AffineExprInfo affineExpr(int64_t coeff, int64_t offset) {
  return {.valid = true, .coeff = coeff, .offset = offset};
}

// Dependency check used only to quickly identify invariant expressions.
static bool dependsOn(Value v, Value vectorIV,
                      llvm::DenseMap<Value, bool> &cache) {
  if (auto it = cache.find(v); it != cache.end())
    return it->second;

  bool result = false;

  if (v == vectorIV) {
    result = true;
  } else if (Operation *op = v.getDefiningOp()) {
    for (Value operand : op->getOperands()) {
      if (dependsOn(operand, vectorIV, cache)) {
        result = true;
        break;
      }
    }
  }

  cache[v] = result;
  return result;
}

// Returns the affine coefficient of vectorIV and folds compile-time constants.
static AffineExprInfo analyzeAffine(Value v, Value vectorIV,
                                    llvm::DenseMap<Value, bool> &depCache) {
  // Vector IV
  if (v == vectorIV)
    return affineCoeff(1);

  // Integer constants
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return affineConstant(attr.getInt());
  }

  // Runtime invariant
  if (!dependsOn(v, vectorIV, depCache))
    return affineExpr(0, 0);

  // Other block arguments are unsupported
  if (isa<BlockArgument>(v))
    return invalidAffine();

  // index_cast
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return analyzeAffine(cast.getIn(), vectorIV, depCache);

  // add
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = analyzeAffine(add.getLhs(), vectorIV, depCache);
    auto rhs = analyzeAffine(add.getRhs(), vectorIV, depCache);

    if (!lhs.valid || !rhs.valid)
      return invalidAffine();

    return affineExpr(lhs.coeff + rhs.coeff, lhs.offset + rhs.offset);
  }

  // sub
  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto lhs = analyzeAffine(sub.getLhs(), vectorIV, depCache);
    auto rhs = analyzeAffine(sub.getRhs(), vectorIV, depCache);

    if (!lhs.valid || !rhs.valid)
      return invalidAffine();

    return affineExpr(lhs.coeff - rhs.coeff, lhs.offset - rhs.offset);
  }

  // mul
  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    auto lhs = analyzeAffine(mul.getLhs(), vectorIV, depCache);
    auto rhs = analyzeAffine(mul.getRhs(), vectorIV, depCache);

    if (!lhs.valid || !rhs.valid)
      return invalidAffine();

    bool lhsDep = lhs.coeff != 0;
    bool rhsDep = rhs.coeff != 0;

    if (lhsDep && rhsDep)
      return invalidAffine();

    // Constant * Constant
    if (!lhsDep && !rhsDep)
      return affineConstant(lhs.offset * rhs.offset);

    // (a*IV+b) * C
    if (lhsDep) {
      return affineExpr(lhs.coeff * rhs.offset, lhs.offset * rhs.offset);
    }

    // C * (a*IV+b)
    return affineExpr(rhs.coeff * lhs.offset, rhs.offset * lhs.offset);
  }

  // Division is non-affine.
  if (isa<arith::DivUIOp, arith::DivSIOp, arith::FloorDivSIOp,
          arith::CeilDivSIOp>(v.getDefiningOp()))
    return invalidAffine();

  // Remainder is non-affine.
  if (isa<arith::RemUIOp, arith::RemSIOp>(v.getDefiningOp()))
    return invalidAffine();

  // Min/max are piecewise affine.
  if (isa<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp>(
          v.getDefiningOp()))
    return invalidAffine();

  // Comparisons are not index expressions.
  if (isa<arith::CmpIOp>(v.getDefiningOp()))
    return invalidAffine();

  // Unknown operation.
  return invalidAffine();
}

enum class AccessKind {
  Invariant,
  Contiguous,
  Strided,
  DynamicStride,
  GatherScatter,
  Unknown
};

struct ResolvedMemrefLayout {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> stride;
  int64_t offset;
};

struct MemrefAccessInfo {
  SmallVector<AffineExprInfo> dims;
  AccessKind kind;
  int varyingDimension = -1;
  int64_t physicalStride = 0;
  ResolvedMemrefLayout layout;
};

// Converts shape to row major strides
void computeRowMajorStrides(SmallVector<int64_t> shape, int N,
                            SmallVector<int64_t> &stride) {
  stride[N - 1] = 1;
  for (int i = N - 2; i >= 0; i--) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
}

std::optional<int64_t> resolveDimensionValue(mlir::Value value) {

  IntegerAttr attr;
  if (matchPattern(value, m_Constant(&attr)))
    return attr.getInt();

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto add = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = resolveDimensionValue(add.getLhs());
    auto rhs = resolveDimensionValue(add.getRhs());

    if (lhs && rhs)
      return *lhs + *rhs;

    return std::nullopt;
  }
  if (auto sub = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = resolveDimensionValue(sub.getLhs());
    auto rhs = resolveDimensionValue(sub.getRhs());

    if (lhs && rhs)
      return *lhs - *rhs;

    return std::nullopt;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhs = resolveDimensionValue(mul.getLhs());
    auto rhs = resolveDimensionValue(mul.getRhs());

    if (lhs && rhs)
      return *lhs * *rhs;

    return std::nullopt;
  }

  if (auto cast = dyn_cast<arith::IndexCastOp>(defOp))
    return resolveDimensionValue(cast.getIn());

  if (auto cast = dyn_cast<arith::IndexCastUIOp>(defOp))
    return resolveDimensionValue(cast.getIn());

  if (auto load = dyn_cast<memref::LoadOp>(defOp)) {
    Value memref = load.getMemRef();
    for (Operation *current = defOp->getPrevNode(); current;
         current = current->getPrevNode()) {
      auto store = dyn_cast<memref::StoreOp>(current);
      if (!store)
        continue;
      // Different memref.
      if (store.getMemRef() != memref)
        continue;

      bool sameIndices = store.getIndices().size() == load.getIndices().size();
      if (sameIndices) {
        for (unsigned i = 0; i < store.getIndices().size(); ++i) {
          if (store.getIndices()[i] != load.getIndices()[i]) {
            sameIndices = false;
            break;
          }
        }
      }
      if (sameIndices)
        return resolveDimensionValue(store.getValueToStore());
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// Produces a concrete shape.
FailureOr<SmallVector<int64_t>> resolveShape(Operation *allocOp) {
  SmallVector<int64_t> resultShape;

  if (allocOp->getNumResults() == 0)
    return failure();
  mlir::Value allocResult = allocOp->getResult(0);

  std::optional<OperandRange> dynamicSizes;
  if (auto alloc = dyn_cast<memref::AllocOp>(allocOp))
    dynamicSizes = alloc.getDynamicSizes();
  else if (auto alloca = dyn_cast<memref::AllocaOp>(allocOp))
    dynamicSizes = alloca.getDynamicSizes();
  else
    return failure();

  int dynamicIndex = 0;

  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(allocResult.getType());
  if (!memrefType)
    return failure();

  if (dynamicSizes.has_value()) {
    OperandRange dynamicSizeVal = dynamicSizes.value();
    unsigned numDims = memrefType.getRank();
    for (unsigned i = 0; i < numDims; ++i) {
      int64_t size = memrefType.getDimSize(i);
      if (size != mlir::ShapedType::kDynamic) {
        resultShape.push_back(size);
      } else {
        mlir::Value dynamicOperand = dynamicSizeVal[dynamicIndex++];
        auto maybeDimValue = resolveDimensionValue(dynamicOperand);
        if (maybeDimValue.has_value()) {
          resultShape.push_back(maybeDimValue.value());
        } else {
          return failure();
        }
      }
    }
  }
  return resultShape;
}

Operation *findAllocation(Value memref) {
  while (true) {
    auto *def = memref.getDefiningOp();
    if (!def)
      return nullptr;

    if (isa<memref::AllocOp, memref::AllocaOp>(def))
      return def;

    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      memref = cast.getSource();
      continue;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      memref = subview.getSource();
      continue;
    }

    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
      memref = rc.getSource();
      continue;
    }

    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      memref = collapse.getSrc();
      continue;
    }

    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      memref = expand.getSrc();
      continue;
    }

    return nullptr;
  }
}

template <typename T> FailureOr<ResolvedMemrefLayout> resolveMemrefShape(T Op) {
  MemRefType memrefType = Op.getMemRefType();
  SmallVector<int64_t> stride;
  int64_t offset;

  // check if Static memref? If yes, return shape
  if (memrefType.hasStaticShape() &&
      !(failed(memrefType.getStridesAndOffset(stride, offset)))) {
    return ResolvedMemrefLayout{
        .shape = llvm::SmallVector<int64_t>(memrefType.getShape()),
        .stride = stride,
        .offset = offset};
  }

  // if not, find defining alloc/alloca
  Operation *allocOp = findAllocation(Op.getMemRef());

  if (!allocOp)
    return failure();

  // resolve each dynamic size operand
  FailureOr<SmallVector<int64_t>> maybeShape = resolveShape(allocOp);
  if (failed(maybeShape)) {
    return failure();
  }
  SmallVector<int64_t> shape = maybeShape.value();

  if (memrefType.getLayout().isIdentity()) {
    int N = shape.size();
    stride.resize(N);
    // compute row-major strides from resolved shape
    computeRowMajorStrides(shape, N, stride);

    return ResolvedMemrefLayout{
        .shape = shape, .stride = stride, .offset = offset};
  }
  return failure();
}

template <typename T>
static MemrefAccessInfo analyzeMemrefAccess(T Op, ValueRange indices,
                                            Value vectorIV) {
  MemrefAccessInfo result;
  llvm::DenseMap<Value, bool> cache;

  for (Value idx : indices) {
    result.dims.push_back(analyzeAffine(idx, vectorIV, cache));
  }

  FailureOr<ResolvedMemrefLayout> maybeLayout = resolveMemrefShape<T>(Op);
  if (failed(maybeLayout)) {
    result.kind = AccessKind::Unknown;
    return result;
  }
  ResolvedMemrefLayout layout = maybeLayout.value();
  result.layout = layout;

  int varying = 0;
  for (unsigned i = 0; i < result.dims.size(); i++) {
    auto &dim = result.dims[i];
    if (!dim.valid) {
      result.kind = AccessKind::Unknown;
      return result;
    }

    if (dim.coeff != 0) {
      varying++;
      result.varyingDimension = i;
      result.physicalStride += dim.coeff * layout.stride[i];
    }
  }

  if (varying == 0) {
    result.kind = AccessKind::Invariant;
  } else if (varying > 1) {
    result.kind = AccessKind::GatherScatter;
  } else if (std::abs(result.physicalStride) == 1) {
    result.kind = AccessKind::Contiguous;
  } else {
    result.kind = AccessKind::Strided;
  }

  return result;
}

struct VectorizationInfo {
  Value mask;
  int vWidth;
  Value pad;
  Type elemType;
  VectorType vType;
  VectorType maskType;
  Value vectorIV;
  DenseMap<Operation *, MemrefAccessInfo> accessInfoMap;
};

struct AccessAnalysis {
  Value vectorIV;
  unsigned vectorIVIndex;
  int score;
  DenseMap<Operation *, MemrefAccessInfo> accessInfoMap;
};

void vectorizeLoads(memref::LoadOp &load, OpBuilder &nestedBuilder,
                    Location nestedLoc, IRMapping &mapping,
                    VectorizationInfo vecInfo);

void vectorizeStores(memref::StoreOp &store, OpBuilder &nestedBuilder,
                     Location nestedLoc, IRMapping &mapping,
                     VectorizationInfo vecInfo);

void vectorizeIfThenElse(scf::IfOp &ifOp, OpBuilder &nestedBuilder,
                         Location nestedLoc, IRMapping &mapping,
                         VectorizationInfo vecInfo);

void vectorizeFor(scf::ForOp &forOp, OpBuilder &nestedBuilder,
                  Location nestedLoc, IRMapping &mapping,
                  VectorizationInfo vecInfo);

void vectorizeOp(Operation &innerOp, OpBuilder &nestedBuilder,
                 Location nestedLoc, IRMapping &mapping,
                 VectorizationInfo vecInfo);

static llvm::StringRef accessKindToString(AccessKind kind) {
  switch (kind) {
  case AccessKind::Invariant:
    return "Invariant";
  case AccessKind::Contiguous:
    return "Contiguous";
  case AccessKind::Strided:
    return "Strided";
  case AccessKind::DynamicStride:
    return "DynamicStride";
  case AccessKind::GatherScatter:
    return "GatherScatter";
  case AccessKind::Unknown:
    return "Unknown";
  }
  return "Invalid";
}

void vectorizeLoads(memref::LoadOp &load, OpBuilder &nestedBuilder,
                    Location nestedLoc, IRMapping &mapping,
                    VectorizationInfo vecInfo) {
  OpBuilder::InsertionGuard guard(nestedBuilder);

  auto it = vecInfo.accessInfoMap.find(load);
  assert(it != vecInfo.accessInfoMap.end());
  const MemrefAccessInfo &result = it->second;

  Value memref = mapping.lookupOrDefault(load.getMemref());
  if (!llvm::isa<MemRefType>(memref.getType()))
    return;

  SmallVector<Value> baseIndices;

  for (unsigned d = 0; d < load.getIndices().size(); ++d) {
    Value idx = mapping.lookupOrDefault(load.getIndices()[d]);

    // The mapped IV is a vector. Gather wants the scalar base.
    if (isa<VectorType>(idx.getType()))
      idx = nestedBuilder.create<vector::ExtractOp>(nestedLoc, idx,
                                                    ArrayRef<int64_t>{0});

    baseIndices.push_back(idx);
  }

  if (result.kind == AccessKind::Contiguous) {
    AffineMap map = AffineMap::get(
        cast<MemRefType>(memref.getType()).getRank(), 0,
        {nestedBuilder.getAffineDimExpr(result.varyingDimension)},
        nestedBuilder.getContext());

    auto readOp = nestedBuilder.create<vector::TransferReadOp>(
        nestedLoc, vecInfo.vType, memref, baseIndices, AffineMapAttr::get(map),
        vecInfo.pad, vecInfo.mask, nestedBuilder.getBoolArrayAttr({false}));

    mapping.map(load.getResult(), readOp.getResult());
  }

  else if (result.kind == AccessKind::Strided ||
           result.kind == AccessKind::DynamicStride) {
    int64_t laneStride = result.layout.stride[result.varyingDimension];

    SmallVector<APInt> offsets;
    offsets.reserve(vecInfo.vWidth);
    for (int64_t lane = 0; lane < vecInfo.vWidth; ++lane)
      offsets.emplace_back(64, lane * laneStride);

    auto idxVecTy =
        VectorType::get({vecInfo.vWidth}, nestedBuilder.getIndexType());

    Value offsetVec = nestedBuilder.create<arith::ConstantOp>(
        nestedLoc, DenseIntElementsAttr::get(idxVecTy, offsets));
    Value passThru = nestedBuilder.create<arith::ConstantOp>(
        nestedLoc, vecInfo.vType, nestedBuilder.getZeroAttr(vecInfo.vType));

    auto gather = nestedBuilder.create<vector::GatherOp>(
        nestedLoc, vecInfo.vType, memref, baseIndices, offsetVec, vecInfo.mask,
        passThru);

    mapping.map(load.getResult(), gather.getResult());
  } else if (result.kind == AccessKind::Invariant) {
    Value scalar =
        nestedBuilder.create<memref::LoadOp>(nestedLoc, memref, baseIndices);
    Value vec = nestedBuilder.create<vector::BroadcastOp>(
        nestedLoc, vecInfo.vType, scalar);
    mapping.map(load.getResult(), vec);

  } else if (result.kind == AccessKind::GatherScatter) {
    llvm_unreachable("Load Gather/scatter not implemented yet");
  } else if (result.kind == AccessKind::Unknown) {
    llvm_unreachable("Load Unknown not implemented yet");
  } else {
    llvm_unreachable("This load is not implemented yet");
  }
}

void vectorizeStores(memref::StoreOp &store, OpBuilder &nestedBuilder,
                     Location nestedLoc, IRMapping &mapping,
                     VectorizationInfo vecInfo) {
  OpBuilder::InsertionGuard guard(nestedBuilder);

  auto it = vecInfo.accessInfoMap.find(store);
  assert(it != vecInfo.accessInfoMap.end());
  const MemrefAccessInfo &result = it->second;

  Value memref = mapping.lookupOrDefault(store.getMemref());
  if (!isa<MemRefType>(memref.getType()))
    return;

  Value value = mapping.lookupOrDefault(store.getValue());

  if (!isa<VectorType>(value.getType()) &&
      result.kind != AccessKind::Invariant) {
    value = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vecInfo.vType,
                                                      value);
  }

  SmallVector<Value> baseIndices;
  for (unsigned d = 0; d < store.getIndices().size(); ++d) {
    Value idx = mapping.lookupOrDefault(store.getIndices()[d]);

    // Scatter wants the scalar base index.
    if (isa<VectorType>(idx.getType()))
      idx = nestedBuilder.create<vector::ExtractOp>(nestedLoc, idx,
                                                    ArrayRef<int64_t>{0});

    baseIndices.push_back(idx);
  }

  if (result.kind == AccessKind::Contiguous) {
    AffineMap map = AffineMap::get(
        cast<MemRefType>(memref.getType()).getRank(), 0,
        {nestedBuilder.getAffineDimExpr(result.varyingDimension)},
        nestedBuilder.getContext());

    nestedBuilder.create<vector::TransferWriteOp>(
        nestedLoc, value, memref, baseIndices, AffineMapAttr::get(map),
        vecInfo.mask, nestedBuilder.getBoolArrayAttr({false}));

    return;
  }

  if (result.kind == AccessKind::Strided ||
      result.kind == AccessKind::DynamicStride) {
    int64_t laneStride = result.layout.stride[result.varyingDimension];

    SmallVector<APInt> offsets;
    offsets.reserve(vecInfo.vWidth);

    for (int64_t lane = 0; lane < vecInfo.vWidth; ++lane)
      offsets.emplace_back(64, lane * laneStride);

    auto idxVecTy =
        VectorType::get({vecInfo.vWidth}, nestedBuilder.getIndexType());
    Value offsetVec = nestedBuilder.create<arith::ConstantOp>(
        nestedLoc, DenseIntElementsAttr::get(idxVecTy, offsets));
    nestedBuilder.create<vector::ScatterOp>(
        nestedLoc, Type(), memref, baseIndices, offsetVec, vecInfo.mask, value);

    return;
  }

  if (result.kind == AccessKind::Invariant) {
    if (isa<VectorType>(value.getType()))
      value = nestedBuilder.create<vector::ExtractOp>(nestedLoc, value,
                                                      ArrayRef<int64_t>{0});

    Value scalarMask = nestedBuilder.create<vector::ExtractOp>(
        nestedLoc, vecInfo.mask, ArrayRef<int64_t>{0});
    nestedBuilder.create<scf::IfOp>(
        nestedLoc, scalarMask, [&](OpBuilder &b, Location loc) {
          b.create<memref::StoreOp>(loc, value, memref, baseIndices);
          b.create<scf::YieldOp>(loc);
        });
    return;
  }

  if (result.kind == AccessKind::GatherScatter)
    llvm_unreachable("Store GatherScatter not implemented yet");

  if (result.kind == AccessKind::Unknown)
    llvm_unreachable("Store Unknown not implemented yet");

  llvm_unreachable("Unhandled store kind");
}

void vectorizeSideEffectingIfThenElse(scf::IfOp &ifOp, OpBuilder &nestedBuilder,
                                      Location nestedLoc, IRMapping &mapping,
                                      VectorizationInfo vecInfo) {

  Value cond = mapping.lookupOrDefault(ifOp.getCondition());

  // Make condition vector-shaped
  if (!isa<VectorType>(cond.getType())) {
    cond = nestedBuilder.create<vector::BroadcastOp>(nestedLoc,
                                                     vecInfo.maskType, cond);
  }
  Value oldMask = vecInfo.mask;

  // -----------------------
  // Then region
  // -----------------------

  Value thenMask =
      nestedBuilder.create<arith::AndIOp>(nestedLoc, oldMask, cond);
  vecInfo.mask = thenMask;
  IRMapping thenMapping(mapping);
  for (Operation &op : ifOp.getThenRegion().front().without_terminator()) {
    vectorizeOp(op, nestedBuilder, nestedLoc, thenMapping, vecInfo);
  }

  // -----------------------
  // Else region
  // -----------------------

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    Value allTrue = nestedBuilder.create<arith::ConstantOp>(
        nestedLoc, DenseElementsAttr::get(vecInfo.maskType, true));
    Value notCond =
        nestedBuilder.create<arith::XOrIOp>(nestedLoc, cond, allTrue);

    Value elseMask =
        nestedBuilder.create<arith::AndIOp>(nestedLoc, oldMask, notCond);
    vecInfo.mask = elseMask;
    IRMapping elseMapping(mapping);

    for (Operation &op : elseRegion.front().without_terminator()) {
      vectorizeOp(op, nestedBuilder, nestedLoc, elseMapping, vecInfo);
    }
  }

  vecInfo.mask = oldMask;
}

void vectorizeIfThenElse(scf::IfOp &ifOp, OpBuilder &nestedBuilder,
                         Location nestedLoc, IRMapping &mapping,
                         VectorizationInfo vecInfo) {

  if (ifOp.getNumResults() == 0) {
    vectorizeSideEffectingIfThenElse(ifOp, nestedBuilder, nestedLoc, mapping,
                                     vecInfo);
  } else {
    auto vectorizeRegion = [](mlir::Region &region, OpBuilder &nestedBuilder,
                              Location nestedLoc, IRMapping &mapping,
                              VectorizationInfo vecInfo) -> SmallVector<Value> {
      if (region.empty())
        return {};

      for (Operation &inner : region.front().without_terminator()) {
        vectorizeOp(inner, nestedBuilder, nestedLoc, mapping, vecInfo);
      }
      SmallVector<Value> result;
      if (auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator())) {
        for (Value v : yield.getOperands()) {
          result.push_back(mapping.lookupOrDefault(v));
        }
      }
      return result;
    };

    Value cond = mapping.lookupOrDefault(ifOp.getCondition());
    if (!llvm::isa<VectorType>(cond.getType())) {
      cond = nestedBuilder.create<vector::BroadcastOp>(nestedLoc,
                                                       vecInfo.maskType, cond);
    }
    Value backup_mask = vecInfo.mask;
    vecInfo.mask =
        nestedBuilder.create<arith::AndIOp>(nestedLoc, backup_mask, cond);
    IRMapping thenMapping(mapping);
    IRMapping elseMapping(mapping);

    auto thenResult = vectorizeRegion(ifOp.getThenRegion(), nestedBuilder,
                                      nestedLoc, thenMapping, vecInfo);

    vecInfo.mask = backup_mask;

    Region &elseRegion = ifOp.getElseRegion();
    if (!elseRegion.empty()) {
      Value allTrue = nestedBuilder.create<arith::ConstantOp>(
          nestedLoc, DenseElementsAttr::get(vecInfo.maskType, true));
      Value notCond =
          nestedBuilder.create<arith::XOrIOp>(nestedLoc, cond, allTrue);
      Value elseMask =
          nestedBuilder.create<arith::AndIOp>(nestedLoc, backup_mask, notCond);
      vecInfo.mask = elseMask;

      auto elseResult = vectorizeRegion(ifOp.getElseRegion(), nestedBuilder,
                                        nestedLoc, elseMapping, vecInfo);

      vecInfo.mask = backup_mask;

      SmallVector<Value> merged;
      for (auto [t, e] : llvm::zip(thenResult, elseResult)) {
        auto ensureVector = [&](Value v) {
          if (isa<VectorType>(v.getType()))
            return v;
          Value val = nestedBuilder.create<vector::BroadcastOp>(
              nestedLoc, VectorType::get({vecInfo.vWidth}, v.getType()), v);
          return val;
        };
        t = ensureVector(t);
        e = ensureVector(e);
        merged.push_back(
            nestedBuilder.create<arith::SelectOp>(nestedLoc, cond, t, e));
      }
      for (auto [orig, newVal] : llvm::zip(ifOp.getResults(), merged)) {
        mapping.map(orig, newVal);
      }
      return;
    }
    vecInfo.mask = backup_mask;
    for (auto [orig, newVal] : llvm::zip(ifOp.getResults(), thenResult)) {
      mapping.map(orig, newVal);
    }
  }
}

void vectorizeFor(scf::ForOp &forOp, OpBuilder &nestedBuilder,
                  Location nestedLoc, IRMapping &mapping,
                  VectorizationInfo vecInfo) {
  SmallVector<Value> vecInitArgs;

  for (Value init : forOp.getInitArgs()) {
    Value mapped = mapping.lookupOrDefault(init);

    if (isa<VectorType>(mapped.getType())) {
      // Already vectorized (e.g. a vector.transfer_read result).
      vecInitArgs.push_back(mapped);
      continue;
    }

    Type t = mapped.getType();
    if (isa<IntegerType, FloatType>(t)) {
      auto vecTy = VectorType::get({vecInfo.vWidth}, t);
      mapped =
          nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vecTy, mapped);
    }

    vecInitArgs.push_back(mapped);
  }

  auto vecFor = nestedBuilder.create<scf::ForOp>(
      nestedLoc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      vecInitArgs);
  Block *newBody = vecFor.getBody();

  nestedBuilder.setInsertionPointToStart(newBody);

  IRMapping bodyMap(mapping);
  bodyMap.map(forOp.getInductionVar(), vecFor.getInductionVar());

  for (auto [oldArg, newArg] :
       llvm::zip(forOp.getRegionIterArgs(), vecFor.getRegionIterArgs())) {
    bodyMap.map(oldArg, newArg);
  }

  for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
    vectorizeOp(bodyOp, nestedBuilder, nestedLoc, bodyMap, vecInfo);
  }

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  SmallVector<Value> yieldVals;
  for (Value val : oldYield.getOperands()) {
    Value mapped = bodyMap.lookup(val);
    yieldVals.push_back(mapped);
  }

  nestedBuilder.setInsertionPointToEnd(vecFor.getBody());

  nestedBuilder.create<scf::YieldOp>(nestedLoc, yieldVals);
  nestedBuilder.setInsertionPointAfter(vecFor);

  for (auto [oldRes, newRes] :
       llvm::zip(forOp.getResults(), vecFor.getResults()))
    mapping.map(oldRes, newRes);
}

void vectorizeOp(Operation &innerOp, OpBuilder &nestedBuilder,
                 Location nestedLoc, IRMapping &mapping,
                 VectorizationInfo vecInfo) {

  if (auto ifOp = dyn_cast<scf::IfOp>(innerOp)) {
    vectorizeIfThenElse(ifOp, nestedBuilder, nestedLoc, mapping, vecInfo);
    return;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(innerOp)) {
    vectorizeFor(forOp, nestedBuilder, nestedLoc, mapping, vecInfo);
    return;
  }

  // Vectorise loads
  if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
    vectorizeLoads(load, nestedBuilder, nestedLoc, mapping, vecInfo);
  }

  // Vectorise stores
  else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
    vectorizeStores(store, nestedBuilder, nestedLoc, mapping, vecInfo);
  }

  // Vectorise Constants safely
  else if (auto constOp = dyn_cast<arith::ConstantOp>(innerOp)) {
    Operation *scalarConst = nestedBuilder.clone(*constOp.getOperation());
    Type scalarType = scalarConst->getResult(0).getType();
    VectorType correctVType = VectorType::get({vecInfo.vWidth}, scalarType);
    Value broadcasted = nestedBuilder.create<vector::BroadcastOp>(
        nestedLoc, correctVType, scalarConst->getResult(0));
    mapping.map(constOp.getResult(), broadcasted);
  }

  // General arith/math vectorisation
  else if (innerOp.getDialect()->getNamespace() == "arith" ||
           innerOp.getDialect()->getNamespace() == "math") {
    SmallVector<Value> operands;
    operands.reserve(innerOp.getNumOperands());

    bool vectorOp = false;

    // First collect mapped operands and determine whether this operation
    // should become vector.
    for (Value operand : innerOp.getOperands()) {
      Value mapped = mapping.lookupOrDefault(operand);

      if (isa<VectorType>(mapped.getType()))
        vectorOp = true;

      operands.push_back(mapped);
    }

    //--------------------------------------------------------------------
    // Pure scalar operation.
    //--------------------------------------------------------------------
    if (!vectorOp) {
      OperationState state(nestedLoc, innerOp.getName().getStringRef());

      state.addOperands(operands);
      state.addTypes(innerOp.getResultTypes());
      state.addAttributes(innerOp.getAttrs());

      Operation *newOp = nestedBuilder.create(state);

      for (auto it : llvm::enumerate(innerOp.getResults()))
        mapping.map(it.value(), newOp->getResult(it.index()));

      return;
    }

    //--------------------------------------------------------------------
    // Vector operation.
    //--------------------------------------------------------------------
    SmallVector<Value> vecOperands;
    vecOperands.reserve(operands.size());

    for (Value v : operands) {
      if (!isa<VectorType>(v.getType())) {
        auto vecTy = VectorType::get({vecInfo.vWidth}, v.getType());
        v = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vecTy, v);
      }
      vecOperands.push_back(v);
    }

    OperationState state(nestedLoc, innerOp.getName().getStringRef());
    state.addOperands(vecOperands);
    for (Type t : innerOp.getResultTypes())
      state.addTypes(VectorType::get({vecInfo.vWidth}, t));

    state.addAttributes(innerOp.getAttrs());
    Operation *newOp = nestedBuilder.create(state);
    for (auto it : llvm::enumerate(innerOp.getResults()))
      mapping.map(it.value(), newOp->getResult(it.index()));
  }
}

int scoreAccess(MemrefAccessInfo info) {
  switch (info.kind) {
  case AccessKind::Contiguous:
    return 100;

  case AccessKind::Strided:
    // int64_t s = std::abs(info.stride);
    return std::max(5, 80 / static_cast<int>(std::abs(info.physicalStride)));

  case AccessKind::Invariant:
    return 10;

  case AccessKind::DynamicStride:
    return 0;

  case AccessKind::GatherScatter:
    return -20;

  case AccessKind::Unknown:
    return -1000;
  }
  return 0;
}

static int64_t estimatedExecutionCount(Operation *op, Operation *parentOp) {
  int64_t weight = 1;
  auto *currentOp = op;

  while (currentOp != parentOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(currentOp)) {
      auto lb = getConstantIntValue(forOp.getLowerBound());
      auto ub = getConstantIntValue(forOp.getUpperBound());
      auto st = getConstantIntValue(forOp.getStep());
      if (!lb.has_value() || !ub.has_value() || !st.has_value())
        return 1;

      int tripCount = (ub.value() - lb.value() + st.value() - 1) / st.value();
      weight *= tripCount;
    }
    currentOp = currentOp->getParentOp();
  }
  return weight;
}

struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
  int targetBitWidth;
  int targetUnrollFactor;

  CUDAToVectorPattern(MLIRContext *ctx, int bitWidth, int unrollFactor)
      : OpRewritePattern<scf::ParallelOp>(ctx), targetBitWidth(bitWidth),
        targetUnrollFactor(unrollFactor) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {

    if (op->hasAttr("vectorized"))
      return failure();
    Location loc = op.getLoc();

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

    // Dynamic vector width calculation
    Type elemType = nullptr;
    op.walk([&](memref::LoadOp load) {
      if (!elemType)
        elemType =
            cast<MemRefType>(load.getMemref().getType()).getElementType();
    });
    if (!elemType)
      elemType = rewriter.getF32Type();

    unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
    int vWidth = targetBitWidth / (elemBitWidth > 0 ? elemBitWidth : 32);
    VectorType vType = VectorType::get({vWidth}, elemType);
    VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
    // targetUnrollFactor = 1;
    Value stepConst = rewriter.create<arith::ConstantIndexOp>(loc, vWidth * 1);

    VectorizationInfo vecInfo;
    vecInfo.maskType = maskType;
    vecInfo.vType = vType;
    vecInfo.vWidth = vWidth;

    auto getReductionKindFromOp =
        [](Operation *mathOp) -> std::optional<vector::CombiningKind> {
      if (isa<arith::AddFOp, arith::AddIOp>(mathOp))
        return vector::CombiningKind::ADD;
      if (isa<arith::MulFOp, arith::MulIOp>(mathOp))
        return vector::CombiningKind::MUL;
      if (isa<arith::MaximumFOp, arith::MaxSIOp>(mathOp))
        return vector::CombiningKind::MAXSI;
      if (isa<arith::MinimumFOp, arith::MinSIOp>(mathOp))
        return vector::CombiningKind::MINSI;
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
              for (Operation &mathOp :
                   llvm::make_early_inc_range(redBlock.getOperations())) {
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
          identityAttr = rewriter.getFloatAttr(
              ft, -std::numeric_limits<double>::infinity());
        else
          identityAttr =
              rewriter.getIntegerAttr(t, std::numeric_limits<int64_t>::min());
      } else {
        identityAttr = rewriter.getZeroAttr(t);
      }
      Value identity = rewriter.create<arith::ConstantOp>(
          loc, cast<TypedAttr>(identityAttr));
      Value broadcasted = rewriter.create<vector::BroadcastOp>(
          loc, VectorType::get({vWidth}, t), identity);
      vectorInits.push_back(broadcasted);
    }

    AccessAnalysis accessAnalysisResult = AnalyzeAccesses(op);

    vecInfo.accessInfoMap = accessAnalysisResult.accessInfoMap;
    vecInfo.vectorIV = accessAnalysisResult.vectorIV;

    rewriter.setInsertionPoint(op);
    SmallVector<Value> lbs(op.getLowerBound());
    SmallVector<Value> ubs(op.getUpperBound());
    SmallVector<Value> steps(op.getStep());

    auto originalIVs = op.getInductionVars();
    steps[accessAnalysisResult.vectorIVIndex] = stepConst;

    auto newLoop = rewriter.create<scf::ParallelOp>(
        loc, lbs, ubs, steps, vectorInits,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs,
            ValueRange /*iterArgs*/) {
          Value zeroIdx =
              nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
          Value ub = op.getUpperBound()[accessAnalysisResult.vectorIVIndex];
          Value pad = nestedBuilder.create<arith::ConstantOp>(
              nestedLoc, elemType, nestedBuilder.getZeroAttr(elemType));
          vecInfo.pad = pad;

          SmallVector<Value> partialSums = vectorInits;

          // for (int u = 0; u < targetUnrollFactor; ++u) {
          for (int u = 0; u < 1; ++u) {
            IRMapping mapping;

            for (unsigned i = 0; i < originalIVs.size(); ++i) {
              mapping.map(originalIVs[i], ivs[i]);
            }

            SmallVector<int64_t> offsets;
            for (int i = 0; i < vWidth; ++i)
              offsets.push_back(i);
            auto tmpVType =
                VectorType::get({vWidth}, nestedBuilder.getIndexType());
            auto laneConst = nestedBuilder.create<arith::ConstantOp>(
                nestedLoc, DenseIntElementsAttr::get(tmpVType, offsets));
            Value offset = nestedBuilder.create<arith::ConstantIndexOp>(
                nestedLoc, u * vWidth);
            Value scalarIV = ivs[accessAnalysisResult.vectorIVIndex];
            Value iv = nestedBuilder.create<arith::AddIOp>(nestedLoc, scalarIV,
                                                           offset);
            Value ivVec = nestedBuilder.create<vector::BroadcastOp>(
                nestedLoc, tmpVType, iv);
            Value laneIV = nestedBuilder.create<arith::AddIOp>(nestedLoc, ivVec,
                                                               laneConst);
            vecInfo.vectorIV = originalIVs[accessAnalysisResult.vectorIVIndex];
            mapping.map(originalIVs[accessAnalysisResult.vectorIVIndex],
                        laneIV);

            Value diff = nestedBuilder.create<arith::SubIOp>(nestedLoc, ub, iv);
            Value clamped =
                nestedBuilder.create<arith::MaxSIOp>(nestedLoc, diff, zeroIdx);
            Value mask = nestedBuilder.create<vector::CreateMaskOp>(
                nestedLoc, maskType, clamped);
            vecInfo.mask = mask;

            for (Operation &innerOp : op.getBody()->getOperations()) {

              // Handle reductions
              if (innerOp.hasTrait<OpTrait::IsTerminator>()) {
                if (auto origReduce = dyn_cast<scf::ReduceOp>(innerOp)) {
                  for (auto [i, operand] :
                       llvm::enumerate(origReduce.getOperands())) {
                    Value val = mapping.lookupOrDefault(operand);
                    if (!llvm::isa<VectorType>(val.getType()))
                      val = nestedBuilder.create<vector::BroadcastOp>(
                          nestedLoc, vType, val);

                    if (llvm::isa<FloatType>(elemType)) {
                      partialSums[i] =
                          (redKind == vector::CombiningKind::MUL)
                              ? nestedBuilder
                                    .create<arith::MulFOp>(nestedLoc,
                                                           partialSums[i], val)
                                    .getResult()
                              : nestedBuilder
                                    .create<arith::AddFOp>(nestedLoc,
                                                           partialSums[i], val)
                                    .getResult();
                    } else {
                      partialSums[i] =
                          (redKind == vector::CombiningKind::MUL)
                              ? nestedBuilder
                                    .create<arith::MulIOp>(nestedLoc,
                                                           partialSums[i], val)
                                    .getResult()
                              : nestedBuilder
                                    .create<arith::AddIOp>(nestedLoc,
                                                           partialSums[i], val)
                                    .getResult();
                    }
                  }
                }
                continue;
              }

              vectorizeOp(innerOp, nestedBuilder, nestedLoc, mapping, vecInfo);
            }
          }

          // Deal with the loop terminator
          auto reduceOp =
              nestedBuilder.create<scf::ReduceOp>(nestedLoc, partialSums);
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
              case vector::CombiningKind::MUL:
                combined =
                    nestedBuilder.create<arith::MulFOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              case vector::CombiningKind::MAXSI:
                combined =
                    nestedBuilder.create<arith::MaximumFOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              case vector::CombiningKind::MINSI:
                combined =
                    nestedBuilder.create<arith::MinimumFOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              default:
                combined =
                    nestedBuilder.create<arith::AddFOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              }
            } else {
              switch (redKind) {
              case vector::CombiningKind::MUL:
                combined =
                    nestedBuilder.create<arith::MulIOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              case vector::CombiningKind::MAXSI:
                combined =
                    nestedBuilder.create<arith::MaxSIOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              case vector::CombiningKind::MINSI:
                combined =
                    nestedBuilder.create<arith::MinSIOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
              default:
                combined =
                    nestedBuilder.create<arith::AddIOp>(nestedLoc, lhs, rhs)
                        .getResult();
                break;
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
        scalarResults.push_back(
            rewriter.create<vector::ReductionOp>(loc, redKind, vRes));
      }
      rewriter.replaceOp(op, scalarResults);
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct CUDAToHierarchicalParallelPass
    : public enzyme::impl::CUDAToHierarchicalParallelBase<
          CUDAToHierarchicalParallelPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    int finalWidth = (regBitWidth > 0) ? regBitWidth : 256;
    int finalUnrollFactor = (unrollFactor > 0) ? unrollFactor : 4;
    // // Added max threads fallback for your tiling pass
    // int maxThreads = 256;

    if (auto attr = module->getAttrOfType<StringAttr>("llvm.target_features")) {
      llvm::StringRef features = attr.getValue();
      if (features.contains("+avx512f"))
        finalWidth = 512;
      else if (features.contains("+avx2") || features.contains("+avx"))
        finalWidth = 256;
      else if (features.contains("+neon") || features.contains("+sse"))
        finalWidth = 128;
    }

    // Move gpu.allocs to memref.allocas
    SmallVector<gpu::AllocOp> allocsToMove;
    module.walk([&](gpu::AllocOp alloc) {
      if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(
              alloc.getType().getMemorySpace())) {
        if (intAttr.getInt() == 3)
          allocsToMove.push_back(alloc);
      }
    });

    for (auto alloc : allocsToMove) {
      func::FuncOp parentFunc = alloc->getParentOfType<func::FuncOp>();
      if (!parentFunc)
        continue;
      OpBuilder hoistedBuilder(&parentFunc.getBody().front(),
                               parentFunc.getBody().front().begin());
      auto stackMem = hoistedBuilder.create<memref::AllocaOp>(
          alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
      alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
      alloc.erase();
    }

    RewritePatternSet patterns(ctx);
    patterns.add<LaunchToParallelPattern>(ctx);
    patterns.add<BarrierFissionPattern>(ctx);
    patterns.add<CUDAToVectorPattern>(ctx, finalWidth, finalUnrollFactor);
    // patterns.add<VectorPattern>(ctx, finalWidth, finalUnrollFactor);
    // patterns.add<ParallelLoopCollapsePattern>(ctx);
    // patterns.add<ParallelLoopTilingPattern>(ctx, finalWidth,
    // finalUnrollFactor, maxThreads);

    // This allows the LaunchToParallelPattern to generate scf.parallel loops,
    // and then immediately feeds those new loops into vectoriser, tiling, and
    // fission patterns.
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
