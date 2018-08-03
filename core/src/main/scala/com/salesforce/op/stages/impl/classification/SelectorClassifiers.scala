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
private[op] trait HasLogisticRegression extends Params
  with SubStage[Stage1ClassificationModelSelector] {
  val sparkLR = new LogisticRegression()

  final val useLR = new BooleanParam(this, "useLR", "boolean to decide to use LogisticRegression in the model selector")
  setDefault(useLR, false)

  private[op] val lRGrid = new ParamGridBuilder()


  /**
   * Logistic Regression Params
   */
  private[op] def setLRParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkLR.getParam(pName).asInstanceOf[Param[T]]
    lRGrid.addGrid(p, values)
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
private[op] trait HasRandomForestClassifier
  extends HasRandomForestBase[ProbClassifier, SelectorClassifiers] {
  override val sparkRF: ProbClassifier = new RandomForestClassifier()
}

/**
 * Decision Tree Classifier For Model Selector
 */
private[op] trait HasDecisionTreeClassifier
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
private[op] trait HasNaiveBayes extends Params with SubStage[Stage1ClassificationModelSelector] {
  val sparkNB = new NaiveBayes()

  final val useNB = new BooleanParam(this, "useNB", "boolean to decide to use NaiveBayes in the model selector")
  setDefault(useNB, false)

  private[op] val nBGrid = new ParamGridBuilder()

  /**
   * Naive Bayes Params
   */

  private[op] def setNBParams[T: ClassTag](pName: String, values: Seq[T]): this.type = {
    val p: Param[T] = sparkNB.getParam(pName).asInstanceOf[Param[T]]
    nBGrid.addGrid(p, values)
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
private[op] trait SelectorClassifiers // TODO add GBT to binary when upgrade to spark 2.2
  extends HasLogisticRegression
    with HasRandomForestClassifier
    with HasDecisionTreeClassifier
    with HasNaiveBayes {

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


  final protected def getModelInfo: Seq[ModelInfo[ProbClassifier]] = Seq(
    ModelInfo(sparkLR.asInstanceOf[ProbClassifier], lRGrid, useLR),
    ModelInfo(sparkRF.asInstanceOf[ProbClassifier], rFGrid, useRF),
    ModelInfo(sparkDT.asInstanceOf[ProbClassifier], dTGrid, useDT),
    ModelInfo(sparkNB.asInstanceOf[ProbClassifier], nBGrid, useNB)
  )

}
