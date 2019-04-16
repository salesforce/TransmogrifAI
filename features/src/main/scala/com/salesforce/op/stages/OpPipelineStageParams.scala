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

package com.salesforce.op.stages

import com.salesforce.op.features._
import com.salesforce.op.features.types.FeatureType
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.{Metadata, StructType}


/**
 * Parameters and functions shared across the input features
 */
trait InputParams extends Params {

  /**
   * FeatureLike should not get serialized with the class since it contains a
   * portion or the entire DAG
   *
   */
  final private[op] val inputFeatures = new TransientFeatureArrayParam(
    parent = this, name = "inputFeatures", doc = "Input features"
  )
  setDefault(inputFeatures, Array[TransientFeature]())

  /**
   * Checks the input length
   *
   * @param features input features
   * @return true is input size as expected, false otherwise
   */
  protected def checkInputLength(features: Array[_]): Boolean


  /**
   * Function to be called on setInput
   */
  protected def onSetInput(): Unit = {}


  /**
   * Sets input features
   *
   * @param features array of input features
   * @tparam S feature like type
   * @return this stage
   */
  final protected def setInputFeatures[S <: OPFeature](features: Array[S]): this.type = {
    require(
      checkInputLength(features),
      "Number of input features must match the number expected by this type of pipeline stage"
    )
    set(inputFeatures, features.map(TransientFeature(_)))
    onSetInput()
    this
  }


  private[op] def setInputFeatureArray[S <: OPFeature](features: Array[S]): this.type = setInputFeatures(features)

  /**
   * Gets the input features
   * Note: this method IS NOT safe to use outside the driver, please use [[getTransientFeatures]] method instead
   *
   * @throws NoSuchElementException if the features are not set
   * @throws RuntimeException       in case one of the features is null
   * @return array of features
   */
  final def getInputFeatures(): Array[OPFeature] = {
    if ($(inputFeatures).isEmpty) throw new NoSuchElementException("Input features are not set")
    else $(inputFeatures).map(_.getFeature())
  }

  /**
   * Gets an input feature
   * Note: this method IS NOT safe to use outside the driver, please use [[getTransientFeature]] method instead
   *
   * @throws NoSuchElementException if the features are not set
   * @throws RuntimeException       in case one of the features is null
   * @return array of features
   */
  final def getInputFeature[T <: FeatureType](i: Int): Option[FeatureLike[T]] = {
    val inputs = getInputFeatures()
    if (inputs.length <= i) None else Option(inputs(i).asInstanceOf[FeatureLike[T]])
  }

  /**
   * Gets the input Features
   *
   * @return input features
   */
  final def getTransientFeatures(): Array[TransientFeature] = $(inputFeatures)

  /**
   * Gets an input feature at index i
   *
   * @param i input index
   * @return maybe an input feature
   */
  final def getTransientFeature(i: Int): Option[TransientFeature] = {
    val inputs = getTransientFeatures()
    if (inputs.length <= i) None else Option(inputs(i))
  }

  /**
   * Input Features type
   */
  type InputFeatures

  /**
   * Function to convert InputFeatures to an Array of FeatureLike
   *
   * @return an Array of FeatureLike
   */
  protected implicit def inputAsArray(in: InputFeatures): Array[OPFeature]
}

/**
 * Parameters shared across all TransmogrifAI base stages
 */
trait OpPipelineStageParams extends InputParams {

  /**
   * Note this should be removed as a param and changed to a var if move stage reader and writer into op
   * and out of ml. Is currently a param to prevent having the setter method be public.
   */
  final private[op] val outputMetadata = new MetadataParam(
    parent = this, name = OpPipelineStageParamsNames.OutputMetadata,
    doc = "any metadata that user wants to save in the transformed DataFrame"
  )

  setDefault(outputMetadata, Metadata.empty)

  final def setMetadata(m: Metadata): this.type = set(outputMetadata, m)

  final def getMetadata(): Metadata = {
    onGetMetadata()
    $(outputMetadata)
  }

  /**
   * Function to be called on getMetadata
   */
  protected def onGetMetadata(): Unit = {}

  /**
   * Note this should be removed as a param and changed to a var if move stage reader and writer into op
   * and out of ml. Is currently a param to prevent having the setter method be public.
   */
  final private[op] val inputSchema = new SchemaParam(
    parent = this, name = OpPipelineStageParamsNames.InputSchema,
    doc = "the schema of the input data from the dataframe"
  )

  setDefault(inputSchema, new StructType())

  final private[op] def setInputSchema(s: StructType): this.type = {
    val featureNames = getInputFeatures().map(_.name)
    val specificSchema = StructType(featureNames.map(s(_)))
    set(inputSchema, specificSchema)
  }

  final def getInputSchema(): StructType = $(inputSchema)

}

object OpPipelineStageParamsNames {
  val OutputMetadata: String = "outputMetadata"
  val InputSchema: String = "inputSchema"
  val InputFeatures: String = "inputFeatures"
}
