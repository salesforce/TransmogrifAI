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

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.OpPipelineStage2
import com.salesforce.op.stages.base.binary.OpTransformer2
import com.salesforce.op.stages.sparkwrappers.generic.SparkWrapperParams
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostRegressionModel}
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import scala.collection.JavaConverters._

import scala.reflect.runtime.universe.TypeTag

/**
 * Wraps a spark ML predictor.  Predictors represent supervised learning algorithms (regression and classification) in
 * spark ML that inherit from [[Predictor]], supported models are:
 * [[org.apache.spark.ml.classification.LogisticRegression]]
 * [[org.apache.spark.ml.regression.LinearRegression]],
 * [[org.apache.spark.ml.classification.RandomForestClassifier]],
 * [[org.apache.spark.ml.regression.RandomForestRegressor]],
 * [[org.apache.spark.ml.classification.NaiveBayesModel]],
 * [[org.apache.spark.ml.classification.GBTClassifier]],
 * [[org.apache.spark.ml.regression.GBTRegressor]],
 * [[org.apache.spark.ml.classification.DecisionTreeClassifier]]
 * [[org.apache.spark.ml.regression.DecisionTreeRegressor]],
 * [[org.apache.spark.ml.classification.LinearSVC]]
 * [[org.apache.spark.ml.classification.MultilayerPerceptronClassifier]],
 * [[org.apache.spark.ml.regression.GeneralizedLinearRegression]].
 * Their defining characteristic is that they output a model which takes in 2 columns as input (labels and features)
 * and output one to three column as result.
 *
 * @param predictor the predictor to wrap
 * @param uid       stage uid
 * @tparam E        spark estimator to wrap
 * @tparam M        spark model returned
 */
class OpPredictorWrapper[E <: Predictor[Vector, E, M], M <: PredictionModel[Vector, M]]
(
  val predictor: E,
  val uid: String = UID[OpPredictorWrapper[_, _]]
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Estimator[OpPredictorWrapperModel[M]] with OpPipelineStage2[RealNN, OPVector, Prediction]
  with SparkWrapperParams[E] {

  val operationName = predictor.getClass.getSimpleName
  val inputParam1Name = SparkMLSharedParamConstants.LabelColName
  val inputParam2Name = SparkMLSharedParamConstants.FeaturesColName
  val outputParamName = SparkMLSharedParamConstants.PredictionColName
  setDefault(sparkMlStage, Option(predictor))

    /**
   * Function that fits the binary model
   */
  override def fit(dataset: Dataset[_]): OpPredictorWrapperModel[M] = {
    setInputSchema(dataset.schema).transformSchema(dataset.schema)
    copyValues(predictor) // when params are shared with wrapping class this will pass them into the model

    val p1 = predictor.getParam(inputParam1Name)
    val p2 = predictor.getParam(inputParam2Name)
    val po = predictor.getParam(outputParamName)
    val model: M = predictor
      .set(p1, in1.name)
      .set(p2, in2.name)
      .set(po, getOutputFeatureName)
      .fit(dataset)

    val wrappedModel = SparkModelConverter.toOP(model, uid)
      .setParent(this)
      .setInput(in1.asFeatureLike[RealNN], in2.asFeatureLike[OPVector])
      .setMetadata(getMetadata())
      .setOutputFeatureName(getOutputFeatureName)

    if (model.isInstanceOf[XGBoostClassificationModel] || model.isInstanceOf[XGBoostRegressionModel]) {
      wrappedModel.setOutputDF(transformFirst(model, dataset))
    }

    wrappedModel
  }

  private def transformFirst(model: Model[_], dataset: Dataset[_]): DataFrame = {
    val first: java.util.List[Row] = List(dataset.toDF().first()).asJava
    val smallDF = SparkSession.active.createDataFrame(first, dataset.schema)
    model.transform(smallDF)
  }
}

abstract class OpPredictorWrapperModel[M <: PredictionModel[Vector, M]]
(
  val operationName: String,
  val uid: String,
  val sparkModel: M
)(
  implicit val tti1: TypeTag[RealNN],
  val tti2: TypeTag[OPVector],
  val tto: TypeTag[Prediction],
  val ttov: TypeTag[Prediction#Value]
) extends Model[OpPredictorWrapperModel[M]] with SparkWrapperParams[M]
  with OpTransformer2[RealNN, OPVector, Prediction] {
  setDefault(sparkMlStage, Option(sparkModel))
}
