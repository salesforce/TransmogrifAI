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

import com.salesforce.op.UID
import com.salesforce.op.aggregators.{Event, FeatureAggregator, GenericFeatureAggregator}
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.features.{Feature, FeatureLike, FeatureUID, OPFeature}
import com.twitter.algebird.MonoidAggregator
import org.apache.spark.ml.PipelineStage
import org.apache.spark.util.ClosureUtils
import org.joda.time.Duration

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Try

/**
 * Origin stage for first features in workflow
 *
 * @param extractFn        function to get data from raw input type
 * @param extractSource    source code of the extract function
 * @param aggregator       type of aggregation to do on feature after extraction
 * @param outputName       name of feature to be returned
 * @param outputIsResponse boolean value of whether response is predictor or response
 *                         (used to determine aggregation window)
 * @param aggregateWindow  time period during which to include features in aggregation
 * @param uid              unique id for stage
 * @param tti              weak type tag for input feature type
 * @param tto              weak type tag for output feature type
 * @tparam I input data type
 * @tparam O output feature type
 */
final class FeatureGeneratorStage[I, O <: FeatureType]
(
  val extractFn: I => O,
  val extractSource: String,
  val aggregator: MonoidAggregator[Event[O], _, O],
  outputName: String,
  override val outputIsResponse: Boolean,
  val aggregateWindow: Option[Duration] = None,
  val uid: String = UID[FeatureGeneratorStage[I, O]]
)(
  implicit val tti: WeakTypeTag[I],
  val tto: WeakTypeTag[O]
) extends PipelineStage with OpPipelineStage[O] with HasIn1 {

  setOutputFeatureName(outputName)

  override type InputFeatures = OPFeature

  override def checkInputLength(features: Array[_]): Boolean = features.length == 0

  override def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  // The output has to be val
  private final val output = Feature[O](
    uid = outputFeatureUid, name = getOutputFeatureName, originStage = this,
    isResponse = outputIsResponse, parents = Seq.empty
  )

  val featureAggregator: FeatureAggregator[I, O, _, O] =
    GenericFeatureAggregator(
      extractFn = extractFn,
      aggregator = aggregator,
      isResponse = outputIsResponse,
      specialTimeWindow = aggregateWindow
    )

  override def getOutput(): FeatureLike[O] = output

  def operationName: String = s"$aggregator($getOutputFeatureName)"

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(extractFn)
}
