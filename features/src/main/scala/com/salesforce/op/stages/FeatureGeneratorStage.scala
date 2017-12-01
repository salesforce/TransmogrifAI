/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
  override val outputName: String,
  override val outputIsResponse: Boolean,
  val aggregateWindow: Option[Duration] = None,
  val uid: String = UID[FeatureGeneratorStage[I, O]]
)(
  implicit val tti: WeakTypeTag[I],
  val tto: WeakTypeTag[O]
) extends PipelineStage with OpPipelineStage[O] with HasIn1 {

  override type InputFeatures = OPFeature

  override def checkInputLength(features: Array[_]): Boolean = features.length == 0

  override def inputAsArray(in: InputFeatures): Array[OPFeature] = Array(in)

  protected[op] override def outputFeatureUid: String = FeatureUID[O](uid)

  // The output has to be val
  private final val output = Feature[O](
    uid = outputFeatureUid, name = outputName, originStage = this, isResponse = outputIsResponse, parents = Seq.empty
  )

  val featureAggregator: FeatureAggregator[I, O, _, O] =
    GenericFeatureAggregator(
      extractFn = extractFn,
      aggregator = aggregator,
      isResponse = outputIsResponse,
      specialTimeWindow = aggregateWindow
    )

  override def getOutput(): FeatureLike[O] = output

  def operationName: String = s"$aggregator($outputName)"

  /**
   * Check if the stage is serializable
   *
   * @return Failure if not serializable
   */
  override def checkSerializable: Try[Unit] = ClosureUtils.checkSerializable(extractFn)
}
