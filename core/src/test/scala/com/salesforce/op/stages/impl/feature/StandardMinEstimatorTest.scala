package com.salesforce.op.stages.impl.feature

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import scala.util.{Failure, Success}

class StandardMinEstimatorTest extends OpEstimatorSpec[Real, UnaryModel[Real, Real], StandardMinEstimator]{

  val inputValues: Seq[Double] = Seq(10, 100, 1000)

  /**
    * Expected result of the transformer applied on the Input Dataset
    */
  override val expectedResult: Seq[Real] = {
    val expectedMean = inputValues.sum / inputValues.length
    val expectedStd = math.sqrt(inputValues.map(value => math.pow(expectedMean - value, 2)).sum
      / (inputValues.length - 1))
    val expectedMin = inputValues.min
    inputValues.map( value => (value - expectedMin) / expectedStd).toReal
  }

  val (inputData, testF) = TestFeatureBuilder(inputValues.map(_.toReal))

  override val estimator: StandardMinEstimator = new StandardMinEstimator().setInput(testF)

  it should "descale and work in standardized positive workflow" in {
    val featureNormalizer = new StandardMinEstimator().setInput(testF)
    val normedOutput = featureNormalizer.getOutput()
    val metadata = featureNormalizer.fit(inputData).getMetadata()

    val expectedMean = inputValues.sum / inputValues.length
    val expectedStd = math.sqrt(inputValues.map(value => math.pow(expectedMean - value, 2)).sum
      / (inputValues.length - 1))
    val expectedMin = inputValues.min
    val expectedSlope = 1 / expectedStd
    val expectedIntercept = - expectedMin / expectedStd
    ScalerMetadata(metadata) match {
      case Failure(err) => fail(err)
      case Success(meta) =>
        meta shouldBe a[ScalerMetadata]
        meta.scalingType shouldBe ScalingType.Linear
        meta.scalingArgs shouldBe a[LinearScalerArgs]
        math.abs((meta.scalingArgs.asInstanceOf[LinearScalerArgs].slope - expectedSlope)
          / expectedSlope) should be < 0.001
        math.abs((meta.scalingArgs.asInstanceOf[LinearScalerArgs].intercept - expectedIntercept)
          / expectedIntercept) should be < 0.001
    }

    val descaledResponse = new DescalerTransformer[Real, Real, Real]()
      .setInput(normedOutput, normedOutput).getOutput()
    val workflow = new OpWorkflow().setResultFeatures(descaledResponse)
    val wfModel = workflow.setInputDataset(inputData).train()
    val transformed = wfModel.score()

    val actual = transformed.collect().map(_.getAs[Double](1))
    val expected : Seq[Double] = inputValues
    all(actual.zip(expected).map(x => math.abs(x._2 - x._1))) should be < 0.0001
  }

}
