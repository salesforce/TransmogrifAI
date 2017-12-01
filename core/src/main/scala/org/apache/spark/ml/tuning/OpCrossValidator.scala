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

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}
import org.json4s.DefaultFormats

/**
 * Modified version of Spark 2.x [[org.apache.spark.ml.tuning.CrossValidator]]
 * (commit d60f6f62d00ffccc40ed72e15349358fe3543311)
 *
 * - Added a boolean param `hasLeakage` to assess label leakage.
 * - In the fit function, data with leakage are grouped by id then split.
 */
class OpCrossValidator (override val uid: String)
  extends Estimator[CrossValidatorModel]
    with CrossValidatorParams with MLWritable with Logging {

  def this() = this(Identifiable.randomUID("cv"))

  private val f2jBLAS = new F2jBLAS

  val hasLeakage: BooleanParam = new BooleanParam(this, "hasLeakage", "if there is leakage")
  setDefault(hasLeakage, false)

  /** @group setParam */
  def setHasLeakage(value: Boolean): this.type = set(hasLeakage, value)

  /** @group setParam */
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  @Since("1.2.0")
  def setNumFolds(value: Int): this.type = set(numFolds, value)

  /** @group setParam */
  @Since("2.0.0")
  def setSeed(value: Long): this.type = set(seed, value)

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {

    val hasLeak = $(hasLeakage)
    val schema = dataset.schema
    transformSchema(schema, logging = true)

    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)
    // scalastyle:off
    import sparkSession.implicits._
    // scalastyle:on
    val rdd = dataset.as[(Double, Vector, Double)].rdd

    val splits = if (hasLeak) {
      // group by ID then split

      val rddRow = rdd.map(p => p._3 -> Row(p._1, p._2))
        .groupByKey()

      MLUtils.kFold(rddRow, $(numFolds), $(seed))
        .map { case (rdd1, rdd2) => (rdd1.values.flatMap(identity), rdd2.values.flatMap(identity))
        }
    } else {
      MLUtils.kFold(rdd.map(p => Row(p._1, p._2)), $(numFolds), $(seed))
    }

    val newSchema = StructType(schema.dropRight(1))


    (splits.zipWithIndex).par.map { case ((training, validation), splitIndex) =>

      val trainingDataset = sparkSession.createDataFrame(training, newSchema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, newSchema).cache()

      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
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
    }
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new CrossValidatorModel(uid, bestModel, metrics).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  override def copy(extra: ParamMap): OpCrossValidator = {
    val copied = defaultCopy(extra).asInstanceOf[OpCrossValidator]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  override def write: MLWriter = new OpCrossValidator.OpCrossValidatorWriter(this)


}

object OpCrossValidator extends MLReadable[OpCrossValidator] {
  override def read: MLReader[OpCrossValidator] = new OpCrossValidatorReader
  override def load(path: String): OpCrossValidator = super.load(path)

  private[OpCrossValidator] class OpCrossValidatorWriter(instance: OpCrossValidator) extends MLWriter {
    ValidatorParams.validateParams(instance)
    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

}

private class OpCrossValidatorReader extends MLReader[OpCrossValidator] {
  private val className = classOf[CrossValidator].getName

  override def load(path: String): OpCrossValidator = {
    implicit val format = DefaultFormats

    val (metadata, estimator, evaluator, estimatorParamMaps) =
      ValidatorParams.loadImpl(path, sc, className)
    val numFolds = (metadata.params \ "numFolds").extract[Int]
    val seed = (metadata.params \ "seed").extract[Long]
    val hasLeakage = (metadata.params \ "hasLeakage").extract[Boolean]
    new OpCrossValidator(metadata.uid)
      .setEstimator(estimator)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(estimatorParamMaps)
      .setNumFolds(numFolds)
      .setSeed(seed)
      .setHasLeakage(hasLeakage)
  }
}
