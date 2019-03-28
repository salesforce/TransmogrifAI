package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.numeric.Number
import com.salesforce.op.utils.tuples.RichTuple._

import scala.reflect.runtime.universe.TypeTag

/**
 * Plus function truth table (Real as example):
 *
 * Real.empty + Real.empty = Real.empty
 * Real.empty + Real(x)    = Real(x)
 * Real(x)    + Real.empty = Real(x)
 * Real(x)    + Real(y)    = Real(x + y)
 **/
class AddTransformer[I1 <: OPNumeric[_], I2 <: OPNumeric[_]]
(
  uid: String = UID[AddTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Real](operationName = "addition", uid = uid){
  override def transformFn: (I1, I2) => Real = (i1: I1, i2: I2) => (i1.toDouble -> i2.toDouble).map(_ + _).toReal
}

/**
 * Scalar addition transformer
 *
 * @param scalar  scalar value
 * @param uid     uid for instance
 * @param tti     type tag for input
 * @param n       value converter
 * @tparam I      input feature type
 * @tparam N      value type
 */
class ScalarAddTransformer[I <: OPNumeric[_], N]
(
  val scalar: N,
  uid: String = UID[ScalarAddTransformer[_, _]]
)(
  implicit override val tti: TypeTag[I],
  n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "scalarAddition", uid = uid){
  override def transformFn: I => Real = (i: I) => i.toDouble.map(_ + n.toDouble(scalar)).toReal
}


/**
 * Minus function truth table (Real as example):
 *
 * Real.empty - Real.empty = Real.empty
 * Real.empty - Real(x)    = Real(-x)
 * Real(x)    - Real.empty = Real(x)
 * Real(x)    - Real(y)    = Real(x - y)
 */
class SubtractTransformer[I1 <: OPNumeric[_], I2 <: OPNumeric[_]]
(
  uid: String = UID[SubtractTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Real](operationName = "minus", uid = uid){
  override def transformFn: (I1, I2) => Real = (i1: I1, i2: I2) => {
    val optZ = (i1.toDouble, i2.toDouble) match {
      case (Some(x), Some(y)) => Some(x - y)
      case (Some(x), None) => Some(x)
      case (None, Some(y)) => Some(-y)
      case (None, None) => None
    }
    optZ.toReal
  }
}


/**
 * Scalar subtract transformer
 *
 * @param scalar   scalar value
 * @param uid      uid for instance
 * @param tti      type tag for input
 * @param n        value converter
 * @tparam I       input feature type
 * @tparam N       value type
 */
class ScalarSubtractTransformer[I <: OPNumeric[_], N]
(
  val scalar: N,
  uid: String = UID[ScalarSubtractTransformer[_, _]]
)(
  implicit override val tti: TypeTag[I],
  n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "scalarSubtract", uid = uid){
  override def transformFn: I => Real = (i: I) => i.toDouble.map(_ - n.toDouble(scalar)).toReal
}

/**
 * Multiply function truth table (Real as example):
 *
 * Real.empty * Real.empty = Real.empty
 * Real.empty * Real(x)    = Real.empty
 * Real(x)    * Real.empty = Real.empty
 * Real(x)    * Real(y)    = Real(x * y) filter ("is not NaN or Infinity")
 */
class MultiplyTransformer[I1 <: OPNumeric[_], I2 <: OPNumeric[_]]
(
  uid: String = UID[MultiplyTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Real](operationName = "multiply", uid = uid){
  override def transformFn: (I1, I2) => Real = (i1: I1, i2: I2) => {
    val result = for {
      x <- i1.toDouble
      y <- i2.toDouble
    } yield x * y

    result filter Number.isValid toReal
  }
}

/**
 * Scalar multiply transformer
 *
 * @param scalar   scalar value
 * @param uid      uid for instance
 * @param tti      type tag for input
 * @param n        value converter
 * @tparam I       input feature type
 * @tparam N       value type
 */
class ScalarMultiplyTransformer[I <: OPNumeric[_], N]
(
  val scalar: N,
  uid: String = UID[ScalarMultiplyTransformer[_, _]]
)(
  implicit override val tti: TypeTag[I],
  n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "scalarMultiply", uid = uid){
  override def transformFn: I => Real = (i: I) => i.toDouble.map(_ * n.toDouble(scalar)).filter(Number.isValid).toReal
}


/**
 * Divide function truth table (Real as example):
 *
 * Real.empty / Real.empty = Real.empty
 * Real.empty / Real(x)    = Real.empty
 * Real(x)    / Real.empty = Real.empty
 * Real(x)    / Real(y)    = Real(x * y) filter ("is not NaN or Infinity")
 */
class DivideTransformer[I1 <: OPNumeric[_], I2 <: OPNumeric[_]]
(
  uid: String = UID[MultiplyTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Real](operationName = "divide", uid = uid){
  override def transformFn: (I1, I2) => Real = (i1: I1, i2: I2) => {
    val result = for {
      x <- i1.toDouble
      y <- i2.toDouble
    } yield x / y

    result filter Number.isValid toReal
  }
}


/**
 * Scalar divide transformer
 *
 * @param scalar   scalar value
 * @param uid      uid for instance
 * @param tti      type tag for input
 * @param n        value converter
 * @tparam I       input feature type
 * @tparam N       value type
 */
class ScalarDivideTransformer[I <: OPNumeric[_], N]
(
  val scalar: N,
  uid: String = UID[ScalarDivideTransformer[_, _]]
)(
  implicit override val tti: TypeTag[I],
  n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "scalarDivide", uid = uid){
  override def transformFn: I => Real = (i: I) => i.toDouble.map(_ / n.toDouble(scalar)).filter(Number.isValid).toReal
}
