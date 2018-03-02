/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.PercentileCalibrator
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpCalibratorReaderWriterTest extends OpPipelineStageReaderWriterTest {
  private val calibrator = new PercentileCalibrator().setInput(height)

  lazy val stage: OpPipelineStageBase = calibrator.fit(passengersDataSet)

  val expected = Array(99.0.toReal, 25.0.toReal, 25.0.toReal, 25.0.toReal, 74.0.toReal, 50.0.toReal)
}
