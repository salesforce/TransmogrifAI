/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.generic

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}

@RunWith(classOf[JUnitRunner])
class SparkWrapperParamsTest extends FlatSpec with BeforeAndAfterEach with TestCommon {

  var swEstimator: SwUnaryEstimator[Real, Real, StandardScalerModel, StandardScaler] = _

  override protected def beforeEach(): Unit = {
    swEstimator = new SwUnaryEstimator[Real, Real, StandardScalerModel, StandardScaler](
      inputParamName = "foo",
      outputParamName = "foo2",
      operationName = "foo3",
      sparkMlStageIn = None
    )
  }

  val path = "test"

  Spec[SparkWrapperParams[_]] should "when setting path, it should also set path to the stage param" in {
    swEstimator.setSavePath(path)

    swEstimator.getSavePath() shouldBe path
    swEstimator.getStageSavePath().get shouldBe swEstimator.getSavePath()
  }

  it should "have proper default values for path and stage" in {
    swEstimator.getSavePath() shouldBe ""
    swEstimator.getSparkMlStage() shouldBe None
  }

  it should "when setting the stage it should also set path" in {
    // should should be none because nothing is set
    swEstimator.getStageSavePath().get shouldBe swEstimator.getSavePath()

    swEstimator.setSavePath(path)
    swEstimator.setSparkMlStage(Some(new StandardScaler()))

    swEstimator.getStageSavePath().get shouldBe swEstimator.getSavePath()
  }
}
