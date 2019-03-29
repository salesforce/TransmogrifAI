/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
 */
class AddTransformer[I1 <: OPNumeric[_], I2 <: OPNumeric[_]]
(
  uid: String = UID[AddTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Real](operationName = "plus", uid = uid){
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
  val n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "plusS", uid = uid){
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
  val n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "minusS", uid = uid){
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
  val n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "multiplyS", uid = uid){
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
  val n: Numeric[N]
) extends UnaryTransformer[I, Real](operationName = "divideS", uid = uid){
  override def transformFn: I => Real = (i: I) => i.toDouble.map(_ / n.toDouble(scalar)).filter(Number.isValid).toReal
}

