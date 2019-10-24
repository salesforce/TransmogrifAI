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
import com.salesforce.op.features.types.{OPVector, Text}
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.functions._

import scala.reflect.runtime.universe.TypeTag

class SmartTextVectorizerWithBias[T <: Text]
(
  uid: String = UID[SmartTextVectorizerWithBias[T]],
  operationName: String = "smartTxtVecWithBias"
)(implicit tti: TypeTag[T]) extends SmartTextVectorizer(
  uid = uid,
  operationName = operationName
) with NameIdentificationFun {

  val defaultThreshold = new DoubleParam(
    parent = this,
    name = "defaultThreshold",
    doc = "default fraction of entries to be names before treating as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(defaultThreshold, 0.50)

  def setThreshold(value: Double): this.type = set(defaultThreshold, value)

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val df = dataset.toDF()
    // TODO: Change this to work correctly on Text#Value rather than Seq[Text#Value]
    val isName = dataset.schema.fieldNames.map { name: String =>
      val column = col(name)
      val (_, treatAsName, _) = estimatorFitFn(df, column, $(defaultThreshold))
      treatAsName
    }

    val modelArgs: SmartTextVectorizerModelArgs = super.fitFn(dataset).asInstanceOf[SmartTextVectorizerModel[T]].args
    val newModelArgs: SmartTextVectorizerModelArgs = modelArgs.copy(isName = isName)

    val newMetadata: OpVectorMetadata = makeVectorMetadata(newModelArgs)
    setMetadata(newMetadata.toMetadata)

    new SmartTextVectorizerModel[T](args = newModelArgs, operationName = operationName, uid = uid)
      .setAutoDetectLanguage(getAutoDetectLanguage)
      .setAutoDetectThreshold(getAutoDetectThreshold)
      .setDefaultLanguage(getDefaultLanguage)
      .setMinTokenLength(getMinTokenLength)
      .setToLowercase(getToLowercase)
      .setTrackTextLen($(trackTextLen))
  }
}
