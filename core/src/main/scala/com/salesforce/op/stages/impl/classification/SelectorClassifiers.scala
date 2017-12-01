/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.stages.impl.classification.ProbabilisticClassifierType.ProbClassifier
import com.salesforce.op.stages.impl.selector._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.ml.tuning.ParamGridBuilder
import enumeratum._

import scala.reflect.ClassTag

/**
 * Enumeration of possible classification models in Model Selector
 */
sealed trait ClassificationModelsToTry extends EnumEntry with Serializable

object ClassificationModelsToTry extends Enum[ClassificationModelsToTry] {
  val values = findValues
  case object LogisticRegression extends ClassificationModelsToTry
  case object RandomForest extends ClassificationModelsToTry
  case object DecisionTree extends ClassificationModelsToTry
  case object NaiveBayes extends ClassificationModelsToTry
}

/**
 * Logistic Regression Classifier for Model Selector
 */
private[classification] trait HasLogisticRegression extends Params
  with SubStage[Stage1ClassificationModelSelector] {
  val sparkLR = new LogisticRegression()

  final val useLR = new BooleanParam(this, "useLR", "boolean to decide to use LogisticRegression in the model selector")
  setDefault(useLR, true)

  private[classification] val lRGrid = new ParamGridBuilder()


  /**
   * Logistic Regression Params
   */
  private[classification] def setLRParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkLR.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkLR.set(p, values.head) else lRGrid.addGrid(p, values)

    subStage.foreach(_.setLRParams[T](pName, values))
    this
  }

  /**
   * Set param for threshold in Logistic Regression prediction, in range [0, 1].
   *
   * @group setParam
   */
  def setLogisticRegressionThreshold(value: Double*): this.type = setLRParams("threshold", value)

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   *
   * @group setParam
   */
  def setLogisticRegressionRegParam(value: Double*): this.type = setLRParams("regParam", value)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * @group setParam
   */
  def setLogisticRegressionElasticNetParam(value: Double*): this.type = setLRParams("elasticNetParam", value)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setLogisticRegressionMaxIter(value: Int*): this.type = setLRParams("maxIter", value)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setLogisticRegressionTol(value: Double*): this.type = setLRParams("tol", value)

  /**
   * Whether to fit an intercept term.
   * Default is true.
   *
   * @group setParam
   */
  def setLogisticRegressionFitIntercept(value: Boolean*): this.type = setLRParams("fitIntercept", value)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users. Note that with/without standardization,
   * the models should be always converged to the same solution when no regularization
   * is applied. In R's GLMNET package, the default behavior is true as well.
   * Default is true.
   *
   * @group setParam
   */
  def setLogisticRegressionStandardization(value: Boolean*): this.type = setLRParams("standardization", value)


}

/**
 * Random Forest Classifier for Model Selector
 */
private[classification] trait HasRandomForestClassifier
  extends HasRandomForestBase[ProbClassifier, SelectorClassifiers] {
  override val sparkRF: ProbClassifier = new RandomForestClassifier()
}

/**
 * Decision Tree Classifier For Model Selector
 */
private[classification] trait HasDecisionTreeClassifier
  extends HasDecisionTreeBase[ProbClassifier, SelectorClassifiers] {
  override val sparkDT: ProbClassifier = new DecisionTreeClassifier()
}


sealed abstract class ModelType(val sparkName: String) extends EnumEntry with Serializable

object ModelType extends Enum[ModelType] {
  val values: Seq[ModelType] = findValues

  case object Multinomial extends ModelType("multinomial")
  case object Bernoulli extends ModelType("bernoulli")
}

/**
 * Naive Bayes Classifier for Model Selector
 */
private[classification] trait HasNaiveBayes extends Params with SubStage[Stage1ClassificationModelSelector] {
  val sparkNB = new NaiveBayes()

  final val useNB = new BooleanParam(this, "useNB", "boolean to decide to use NaiveBayes in the model selector")
  setDefault(useNB, false)

  private[impl] val nBGrid = new ParamGridBuilder()

  /**
   * Naive Bayes Params
   */

  private[impl] def setNBParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkNB.getParam(pName).asInstanceOf[Param[T]]
    if (values.distinct.length == 1) sparkNB.set(p, values.head) else nBGrid.addGrid(p, values)

    subStage.map(_.setNBParams[T](pName, values))
    this
  }

  /**
   * Set the smoothing parameter.
   * Default is 1.0.
   *
   * @group setParam
   */
  def setNaiveBayesSmoothing(value: Double*): this.type = setNBParams("smoothing", value)

  /**
   * Set the model type using a string (case-sensitive).
   * Supported options: "multinomial" and "bernoulli".
   * Default is "multinomial"
   *
   * @group setParam
   */
  def setNaiveBayesModelType(value: ModelType*): this.type = setNBParams("modelType", value.map(_.sparkName))
}


/**
 * Classifiers to try in the Model Selector
 */
private[impl] trait SelectorClassifiers
  extends HasLogisticRegression
    with HasRandomForestClassifier
    with HasDecisionTreeClassifier
    with HasNaiveBayes
    with SelectorModels[ProbClassifier, Stage1ClassificationModelSelector]
    with Stage3ParamNamesBase {

  /**
   * Set param for Thresholds of all classifiers to adjust the probability of predicting each class.
   * Array must have length equal to the number of classes, with values >= 0. The class with largest value p/t is
   * predicted, where p is the original probability of that class and t is the class' threshold.
   *
   * The thresholds should be the same for all classifiers
   *
   *
   * @group setParam
   */
  def setModelThresholds(value: Array[Double]): this.type = {
    sparkLR.setThresholds(value)
    sparkRF.asInstanceOf[RandomForestClassifier].setThresholds(value)
    sparkDT.asInstanceOf[DecisionTreeClassifier].setThresholds(value)
    sparkNB.setThresholds(value)
    subStage.foreach(_.setModelThresholds(value: Array[Double]))
    this
  }

  /**
   * Get thresholds
   *
   * @group getParam
   */
  def getThresholds: Array[Double] = sparkLR.getThresholds

  // scalastyle:off
  import ClassificationModelsToTry._

  // scalastyle:on

  /**
   * Set the models to try for the model selector.
   * The models can be LogisticRegression, RandomForest, DecisionTree and NaiveBayes
   *
   * @group setParam
   */
  def setModelsToTry(modelsToTry: ClassificationModelsToTry*): this.type = {
    val potentialModelSet = modelsToTry.toSet
    set(useLR, potentialModelSet(LogisticRegression))
    set(useRF, potentialModelSet(RandomForest))
    set(useDT, potentialModelSet(DecisionTree))
    set(useNB, potentialModelSet(NaiveBayes))
    subStage.foreach(_.setModelsToTry(modelsToTry: _*))
    this
  }


  final override protected[impl] def modelInfo: Seq[ModelInfo[ProbClassifier]] = Seq(
    ModelInfo(sparkLR.asInstanceOf[ProbClassifier], lRGrid, useLR),
    ModelInfo(sparkRF.asInstanceOf[ProbClassifier], rFGrid, useRF),
    ModelInfo(sparkDT.asInstanceOf[ProbClassifier], dTGrid, useDT),
    ModelInfo(sparkNB.asInstanceOf[ProbClassifier], nBGrid, useNB)
  )
}
