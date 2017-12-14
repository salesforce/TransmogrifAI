/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.iris

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.readers.CustomReader
import com.salesforce.op.stages.impl.classification.{ClassificationModelsToTry, Impurity, MultiClassificationModelSelector}
import com.salesforce.op.stages.impl.tuning.DataSplitter
import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}

/**
 * Optimus Prime MultiClass Classification example on the Iris Dataset
 */
object OpIris extends OpAppWithRunner with IrisFeatures {

  override def kryoRegistrator: Class[_ <: OpKryoRegistrator] = classOf[IrisKryoRegistrator]

  ////////////////////////////////////////////////////////////////////////////////
  // READER DEFINITIONS
  /////////////////////////////////////////////////////////////////////////////////

  val randomSeed = 112233

  val irisReader = new CustomReader[Iris](key = _.getID.toString){
    def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[Iris], Dataset[Iris]] = {
      val path = getFinalReadPath(params)
      val myFile = spark.sparkContext.textFile(path)

      Left(myFile.filter(_.nonEmpty).zipWithIndex.map { case (x, number) =>
        val words = x.split(",")
        new Iris(number.toInt, words(0).toDouble, words(1).toDouble, words(2).toDouble, words(3).toDouble, words(4))
      })
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // WORKFLOW DEFINITION
  /////////////////////////////////////////////////////////////////////////////////

  private val labels = irisClass.indexed()

  private val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()

  val (pred, raw, prob) = MultiClassificationModelSelector
    .withCrossValidation(splitter = Some(DataSplitter(reserveTestFraction = 0.2, seed = randomSeed)), seed = randomSeed)
    .setModelsToTry(ClassificationModelsToTry.LogisticRegression, ClassificationModelsToTry.DecisionTree)
    .setLogisticRegressionMaxIter(10, 100)
    .setLogisticRegressionRegParam(0.01, 0.1)
    .setDecisionTreeMaxDepth(10, 20, 30)
    .setDecisionTreeImpurity(Impurity.Gini)
    .setDecisionTreeSeed(randomSeed)
    .setInput(labels, features).getOutput()

  private val evaluator = Evaluators.MultiClassification.f1()
    .setLabelCol(labels)
    .setPredictionCol(pred)
    .setRawPredictionCol(raw)
    .setProbabilityCol(prob)

  private val wf = new OpWorkflow().setResultFeatures(pred, raw, prob, labels)

  def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = wf,
      trainingReader = irisReader,
      scoringReader = irisReader,
      evaluationReader = Option(irisReader),
      evaluator = evaluator,
      featureToComputeUpTo = features
    )
}
