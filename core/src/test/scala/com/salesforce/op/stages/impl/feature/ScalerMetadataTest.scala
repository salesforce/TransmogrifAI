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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.json.JsonUtils
import org.apache.spark.sql.types.MetadataBuilder

import scala.util.{Failure, Success}

@RunWith(classOf[JUnitRunner])
class ScalerMetadataTest extends FlatSpec with TestSparkContext {
  val linearArgs = LinearScalerArgs(slope = 2.0, intercept = 1.0)

  Spec[ScalerMetadata] should "properly construct ScalerMetadata for a LinearScaler" in {
    val metadata = ScalerMetadata(scalingType = ScalingType.Linear,
      scalingArgs = linearArgs).toMetadata()
    metadata.getString(ScalerMetadata.scalingTypeName) shouldBe ScalingType.Linear.entryName
    val args = JsonUtils.fromString[LinearScalerArgs](metadata.getString(ScalerMetadata.scalingArgsName))
    args match {
      case Failure(err) => fail(err)
      case Success(x) => x shouldBe linearArgs
    }
  }

  it should "properly construct ScalerMetaData for a LogScaler" in {
    val metadata = ScalerMetadata(scalingType = ScalingType.Logarithmic, scalingArgs = EmptyArgs()).toMetadata()
    metadata.getString(ScalerMetadata.scalingTypeName) shouldBe ScalingType.Logarithmic.entryName
    metadata.getString(ScalerMetadata.scalingArgsName) shouldBe "{}"
  }

  it should "use apply to properly convert metadata to ScalerMetadata" in {
    val metadata = new MetadataBuilder().putString(ScalerMetadata.scalingTypeName, ScalingType.Linear.entryName)
      .putString(ScalerMetadata.scalingArgsName, linearArgs.toJson(pretty = false)).build()
    ScalerMetadata.apply(metadata) match {
      case Failure(err) => fail(err)
      case Success(x) => x shouldBe ScalerMetadata(ScalingType.Linear, linearArgs)
    }
  }

  it should "error when apply is given an invalid scaling type" in {
    val invalidMetaData = new MetadataBuilder().putString(ScalerMetadata.scalingTypeName, "unsupportedScaling")
      .putString(ScalerMetadata.scalingArgsName, linearArgs.toJson(pretty = false)).build()

    val err = intercept[NoSuchElementException] (
      ScalerMetadata.apply(invalidMetaData).get
    )
    err.getMessage shouldBe "unsupportedScaling is not a member of Enum (Linear, Logarithmic)"
  }
}
