package com.salesforce.op.stages.impl.feature

import com.salesforce.op.{OpWorkflow, OpWorkflowModelWriter}
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.features.types.Real
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

@RunWith(classOf[JUnitRunner])
class  ScalerTransformerTest extends FlatSpec with TestSparkContext {
  val (testData, inA) = TestFeatureBuilder("inA", Seq[(Real)](Real(4.0), Real(1.0), Real(0.0)))

  Spec[ScalerTransformer[_, _]] should "Properly linearly scale numeric fields" in {
    val predScaler = new ScalerTransformer[Real, Real](
      scalingType = ScalingType.Linear,
      scalingArgs = LinearScalerArgs(slope = 2.0, intercept = 1.0)
    ).setInput(inA.asInstanceOf[Feature[Real]])
    val vec: FeatureLike[Real] = predScaler.getOutput()
    val wfModel = new OpWorkflow().setResultFeatures(vec).setInputDataset(testData).train()
    val saveModelPath = tempDir.toPath.toString + "linearScalerTest" + DateTime.now().getMillis.toString
    OpWorkflowModelWriter.toJson(wfModel, saveModelPath)
    val data = wfModel.score()
    val actual = data.collect()
    actual.map(_.getAs[Double](1)) shouldEqual Array(9.0, 3.0, 1.0)
    val metadata = data.schema(vec.name).metadata
    metadata.getString("scalingType") shouldEqual ScalingType.Linear.entryName
    val scalingArgs = JsonUtils.fromString[LinearScalerArgs](metadata.getString("scalingArgs")).get
    scalingArgs.intercept shouldEqual 1.0
    scalingArgs.slope shouldEqual 2.0
  }
}