/*
 * Modifications: (c) 2017, Salesforce.com, Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package org.apache.spark.ml.tuning

import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}
import org.json4s.DefaultFormats

import scala.language.existentials

/**
 * Modified version of Spark 2.x [[org.apache.spark.ml.tuning.TrainValidationSplit]]
 * (commit d60f6f62d00ffccc40ed72e15349358fe3543311)
 * In the fit function, data is grouped by key then split to avoid leakage
 */
class OpTrainValidationSplit (override val uid: String)
  extends Estimator[TrainValidationSplitModel]
    with TrainValidationSplitParams with MLWritable with Logging {

  def this() = this(Identifiable.randomUID("tvs"))

  /** @group setParam */
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  def setTrainRatio(value: Double): this.type = set(trainRatio, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  override def fit(dataset: Dataset[_]): TrainValidationSplitModel = {

    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)

    import dataset.sparkSession.implicits._
    val rdd = dataset.as[LabelFeaturesKey].rdd
    // group by ID then split
    val Array(trainingRDD, validationRDD) = rdd.map(p => p._3 -> Row(p._1, p._2))
      .groupByKey().randomSplit(Array($(trainRatio), 1 - $(trainRatio)), $(seed)).map(_.values.flatMap(identity))

    val sparkSession = dataset.sparkSession
    val newSchema = StructType(schema.dropRight(1))
    val trainingDataset = sparkSession.createDataFrame(trainingRDD, newSchema).cache()
    val validationDataset = sparkSession.createDataFrame(validationRDD, newSchema).cache()

    // multi-model training
    logDebug(s"Train split with multiple sets of parameters.")
    val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
    trainingDataset.unpersist()
    var i = 0
    while (i < numModels) {
      val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
      logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
      metrics(i) += metric
      i += 1
    }
    validationDataset.unpersist()

    logInfo(s"Train validation split metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best train validation split metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new TrainValidationSplitModel(uid, bestModel, metrics).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  override def copy(extra: ParamMap): OpTrainValidationSplit = {
    val copied = defaultCopy(extra).asInstanceOf[OpTrainValidationSplit]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  override def write: MLWriter = new OpTrainValidationSplit.OpTrainValidationSplitWriter(this)
}

object OpTrainValidationSplit extends MLReadable[OpTrainValidationSplit] {
  override def read: MLReader[OpTrainValidationSplit] = new OpTrainValidationSplitReader
  override def load(path: String): OpTrainValidationSplit = super.load(path)

  private[OpTrainValidationSplit] class OpTrainValidationSplitWriter
  (
    instance: OpTrainValidationSplit
  ) extends MLWriter {
    ValidatorParams.validateParams(instance)
    override protected def saveImpl(path: String): Unit = ValidatorParams.saveImpl(path, instance, sc)
  }

  private class OpTrainValidationSplitReader extends MLReader[OpTrainValidationSplit] {
    private val className = classOf[OpTrainValidationSplit].getName

    override def load(path: String): OpTrainValidationSplit = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val trainRatio = (metadata.params \ "trainRatio").extract[Double]
      val seed = (metadata.params \ "seed").extract[Long]
      new OpTrainValidationSplit(metadata.uid)
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(estimatorParamMaps)
        .setTrainRatio(trainRatio)
        .setSeed(seed)
    }
  }

}
