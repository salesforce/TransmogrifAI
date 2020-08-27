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

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, Prediction, RealNN}
import com.salesforce.op.stages.impl.CheckIsResponseValues
import com.salesforce.op.stages.sparkwrappers.specific.{OpPredictorWrapper, OpProbabilisticClassifierModel}
import ml.combust.mleap.xgboost.runtime.{XGBoostClassificationModel => MleapXGBoostClassificationModel}
import ml.dmlc.xgboost4j.scala.spark._
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.linalg.Vectors

import scala.reflect.runtime.universe._

/**
 * Wrapper around XGBoost classifier [[XGBoostClassifier]]
 */
class OpXGBoostClassifier(uid: String = UID[OpXGBoostClassifier])
  extends OpPredictorWrapper[XGBoostClassifier, XGBoostClassificationModel](
    predictor = new XGBoostClassifier(),
    uid = uid
  ) with OpXGBoostClassifierParams {

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    CheckIsResponseValues(in1, in2)
  }

  /**
   * Weight column name. If this is not set or empty, we treat all instance weights as 1.0.
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
   * Initial prediction (aka base margin) column name.
   */
  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  /**
   * Number of classes
   */
  def setNumClass(value: Int): this.type = set(numClass, value)

  // setters for general params

  /**
   * Rabit tracker configurations. The parameter must be provided as an instance of the
   * [[TrackerConf]] class, which has the following definition:
   *
   * case class TrackerConf(workerConnectionTimeout: Duration, trainingTimeout: Duration, trackerImpl: String)
   *
   * See below for detailed explanations.
   *
   *   - trackerImpl: Select the implementation of Rabit tracker.
   * default: "python"
   *
   * Choice between "python" or "scala". The former utilizes the Java wrapper of the
   * Python Rabit tracker (in dmlc_core), and does not support timeout settings.
   * The "scala" version removes Python components, and fully supports timeout settings.
   *
   *   - workerConnectionTimeout: the maximum wait time for all workers to connect to the tracker.
   * default: 0 millisecond (no timeout)
   *
   * The timeout value should take the time of data loading and pre-processing into account,
   * due to the lazy execution of Spark's operations. Alternatively, you may force Spark to
   * perform data transformation before calling XGBoost.train(), so that this timeout truly
   * reflects the connection delay. Set a reasonable timeout value to prevent model
   * training/testing from hanging indefinitely, possible due to network issues.
   * Note that zero timeout value means to wait indefinitely (equivalent to Duration.Inf).
   * Ignored if the tracker implementation is "python".
   */
  def setTrackerConf(value: TrackerConf): this.type = set(trackerConf, value)

  /**
   * The number of rounds for boosting
   */
  def setNumRound(value: Int): this.type = set(numRound, value)

  /**
   * Number of workers used to train xgboost model. default: 1
   */
  def setNumWorkers(value: Int): this.type = set(numWorkers, value)

  /**
   * Number of threads used by per worker. default 1
   */
  def setNthread(value: Int): this.type = set(nthread, value)

  /**
   * Whether to use external memory as cache. default: false
   */
  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  /**
   * 0 means printing running messages, 1 means silent mode. default: 0
   */
  def setSilent(value: Int): this.type = set(silent, value)

  /**
   * The value treated as missing
   */
  def setMissing(value: Float): this.type = set(missing, value)

  /**
   * The maximum time to wait for the job requesting new workers. default: 30 minutes
   */
  def setTimeoutRequestWorkers(value: Long): this.type = set(timeoutRequestWorkers, value)

  /**
   * The hdfs folder to load and save checkpoint boosters. default: `empty_string`
   */
  def setCheckpointPath(value: String): this.type = set(checkpointPath, value)

  /**
   * Checkpoint interval (&gt;= 1) or disable checkpoint (-1). E.g. 10 means that
   * the trained model will get checkpointed every 10 iterations. Note: `checkpoint_path` must
   * also be set if the checkpoint interval is greater than 0.
   */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /**
   * Random seed for the C++ part of XGBoost and train/test splitting.
   */
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Step size shrinkage used in update to prevents overfitting. After each boosting step, we
   * can directly get the weights of new features and eta actually shrinks the feature weights
   * to make the boosting process more conservative. [default=0.3] range: [0,1]
   */
  def setEta(value: Double): this.type = set(eta, value)

  /**
   * Minimum loss reduction required to make a further partition on a leaf node of the tree.
   * the larger, the more conservative the algorithm will be. [default=0] range: [0,
   * Double.MaxValue]
   */
  def setGamma(value: Double): this.type = set(gamma, value)

  /**
   * Maximum depth of a tree, increase this value will make model more complex / likely to be
   * overfitting. [default=6] range: [1, Int.MaxValue]
   */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /**
   * Minimum sum of instance weight(hessian) needed in a child. If the tree partition step results
   * in a leaf node with the sum of instance weight less than min_child_weight, then the building
   * process will give up further partitioning. In linear regression mode, this simply corresponds
   * to minimum number of instances needed to be in each node. The larger, the more conservative
   * the algorithm will be. [default=1] range: [0, Double.MaxValue]
   */
  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  /**
   * Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it
   * means there is no constraint. If it is set to a positive value, it can help making the update
   * step more conservative. Usually this parameter is not needed, but it might help in logistic
   * regression when class is extremely imbalanced. Set it to value of 1-10 might help control the
   * update. [default=0] range: [0, Double.MaxValue]
   */
  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  /**
   * Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly
   * collected half of the data instances to grow trees and this will prevent overfitting.
   * [default=1] range:(0,1]
   */
  def setSubsample(value: Double): this.type = set(subsample, value)

  /**
   * Subsample ratio of columns when constructing each tree. [default=1] range: (0,1]
   */
  def setColsampleBytree(value: Double): this.type = set(colsampleBytree, value)

  /**
   * Subsample ratio of columns for each split, in each level. [default=1] range: (0,1]
   */
  def setColsampleBylevel(value: Double): this.type = set(colsampleBylevel, value)

  /**
   * L2 regularization term on weights, increase this value will make model more conservative.
   * [default=1]
   */
  def setLambda(value: Double): this.type = set(lambda, value)

  /**
   * L1 regularization term on weights, increase this value will make model more conservative.
   * [default=0]
   */
  def setAlpha(value: Double): this.type = set(alpha, value)

  /**
   * The tree construction algorithm used in XGBoost. options: {'auto', 'exact', 'approx'}
   *  [default='auto']
   */
  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  /**
   * Growth policy for fast histogram algorithm
   */
  def setGrowPolicy(value: String): this.type = set(growPolicy, value)

  /**
   * Maximum number of bins in histogram
   */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /**
   * Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
   */
  def setMaxLeaves(value: Int): this.type = set(maxLeaves, value)

  /**
   * This is only used for approximate greedy algorithm.
   * This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select
   * number of bins, this comes with theoretical guarantee with sketch accuracy.
   * [default=0.03] range: (0, 1)
   */
  def setSketchEps(value: Double): this.type = set(sketchEps, value)

  /**
   * Control the balance of positive and negative weights, useful for unbalanced classes. A typical
   * value to consider: sum(negative cases) / sum(positive cases).   [default=1]
   */
  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)

  /**
   * Parameter for Dart booster.
   * Type of sampling algorithm. "uniform": dropped trees are selected uniformly.
   * "weighted": dropped trees are selected in proportion to weight. [default="uniform"]
   */
  def setSampleType(value: String): this.type = set(sampleType, value)

  /**
   * Parameter of Dart booster.
   * type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
   */
  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  /**
   * Parameter of Dart booster.
   * dropout rate. [default=0.0] range: [0.0, 1.0]
   */
  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  /**
   * Parameter of Dart booster.
   * probability of skip dropout. If a dropout is skipped, new trees are added in the same manner
   * as gbtree. [default=0.0] range: [0.0, 1.0]
   */
  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  /**
   * Parameter of linear booster
   * L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)
   */
  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  // setters for learning params

  /**
   * Specify the learning task and the corresponding learning objective.
   * options: reg:squarederror, reg:logistic, binary:logistic, binary:logitraw, count:poisson,
   * multi:softmax, multi:softprob, rank:pairwise, reg:gamma. default: reg:squarederror
   */
  def setObjective(value: String): this.type = set(objective, value)

  /**
   * Objective type used for training. For options see [[ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams]]
   */
  def setObjectiveType(value: String): this.type = set(objectiveType, value)

  /**
   * Specify the learning task and the corresponding learning objective.
   * options: reg:linear, reg:logistic, binary:logistic, binary:logitraw, count:poisson,
   * multi:softmax, multi:softprob, rank:pairwise, reg:gamma. default: reg:linear
   */
  def setBaseScore(value: Double): this.type = set(baseScore, value)

  /**
   * Evaluation metrics for validation data, a default metric will be assigned according to
   * objective(rmse for regression, and error for classification, mean average precision for
   * ranking). options: rmse, mae, logloss, error, merror, mlogloss, auc, aucpr, ndcg, map,
   * gamma-deviance
   */
  def setEvalMetric(value: String): this.type = set(evalMetric, value)

  /**
   * Fraction of training points to use for testing.
   */
  def setTrainTestRatio(value: Double): this.type = set(trainTestRatio, value)

  /**
   * If non-zero, the training will be stopped after a specified number
   * of consecutive increases in any evaluation metric.
   */
  def setNumEarlyStoppingRounds(value: Int): this.type = set(numEarlyStoppingRounds, value)

  /**
   * Define the expected optimization to the evaluation metrics, true to maximize otherwise minimize it
   */
  def setMaximizeEvaluationMetrics(value: Boolean): this.type = set(maximizeEvaluationMetrics, value)

  /**
   * Customized objective function provided by user. default: null
   */
  def setCustomObj(value: ObjectiveTrait): this.type = set(customObj, value)

  /**
   * Customized evaluation function provided by user. default: null
   */
  def setCustomEval(value: EvalTrait): this.type = set(customEval, value)

}


/**
 * Class that takes in a spark [[XGBoostClassificationModel]] and wraps it into an OP model which returns a
 * Prediction feature
 *
 * @param sparkModel    model to wrap
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid to give stage
 */
class OpXGBoostClassificationModel
(
  sparkModel: XGBoostClassificationModel,
  uid: String = UID[OpXGBoostClassificationModel],
  operationName: String = classOf[XGBoostClassifier].getSimpleName
)(
  implicit tti1: TypeTag[RealNN],
  tti2: TypeTag[OPVector],
  tto: TypeTag[Prediction],
  ttov: TypeTag[Prediction#Value]
) extends OpProbabilisticClassifierModel[XGBoostClassificationModel](
  sparkModel = sparkModel, uid = uid, operationName = operationName
) {
  import OpXGBoost._

  protected def predictRawMirror: MethodMirror =
    throw new NotImplementedError(
      "XGBoost-Spark does not support 'predictRaw'. This might change in upcoming releases.")

  protected def raw2probabilityMirror: MethodMirror =
    throw new NotImplementedError(
      "XGBoost-Spark does not support 'raw2probability'. This might change in upcoming releases.")

  @transient lazy val probability2predictionMirror = getSparkOrLocalMethod("probability2prediction",
    "probabilityToPrediction")

  private lazy val (booster: Booster, treeLim: Int, missing: Float, numClasses: Int) = {
    getSparkMlStage()
      .map{ model => (model.nativeBooster, model.getTreeLimit, model.getMissing, model.numClasses) }
      .orElse{
        getLocalMlStage().map(_.model.asInstanceOf[MleapXGBoostClassificationModel])
          .map{ model => (model.booster, model.treeLimit, Float.NaN, model.numClasses) }
      }.getOrElse( throw new RuntimeException("Could not find spark or local wrapped XGBoost") )
  }

  override def transformFn: (RealNN, OPVector) => Prediction = (_, features) => {
    val data = processMissingValues(Iterator(features.value.asXGB), missing)
    val dm = new DMatrix(dataIter = data)
    val rawPred = booster.predict(dm, outPutMargin = true, treeLimit = treeLim)(0).map(_.toDouble)
    val rawPrediction = if (numClasses == 2) Array(-rawPred(0), rawPred(0)) else rawPred
    val prob = booster.predict(dm, outPutMargin = false, treeLimit = treeLim)(0).map(_.toDouble)
    val probability = if (numClasses == 2) Array(1.0 - prob(0), prob(0)) else prob
    val prediction = probability2predictionMirror(Vectors.dense(probability)).asInstanceOf[Double]

    Prediction(prediction = prediction, rawPrediction = rawPrediction, probability = probability)
  }
}
