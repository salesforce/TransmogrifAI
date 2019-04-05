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
import org.apache.spark.ml.param.{BooleanParam, Params}

import scala.reflect.runtime.universe.TypeTag

/**
 * Checks if the first input is a substring of the second input
 * @param uid           uid for instance
 * @param tti1          type tag for first input
 * @param tti2          type tag for second input
 * @tparam I1 first input feature type
 * @tparam I2 second input feature type
 */
class SubstringTransformer[I1 <: Text, I2 <: Text](
  uid: String = UID[SubstringTransformer[_, _]]
)(
  implicit override val tti1: TypeTag[I1],
  override val tti2: TypeTag[I2]
) extends BinaryTransformer[I1, I2, Binary](operationName = "substring", uid = uid) with TextMatchingParams {
  override def transformFn: (I1, I2) => Binary = (sub: I1, full: I2) => {
    val (subClean, fullClean) =
      if ($(toLowercase)) (sub.map(_.toLowerCase), full.map(_.toLowerCase))
      else (sub.value, full.value)
    fullClean.flatMap(f => subClean.map(f.contains(_))).toBinary
  }
}


trait TextMatchingParams extends Params {

  /**
   * Indicates whether to convert all characters to lowercase before string operation.
   */
  final val toLowercase =
    new BooleanParam(this, "toLowercase", "whether to convert all characters to lowercase before string operation")
  def setToLowercase(value: Boolean): this.type = set(toLowercase, value)
  def getToLowercase: Boolean = $(toLowercase)
  setDefault(toLowercase -> TextTokenizer.ToLowercase)

}
