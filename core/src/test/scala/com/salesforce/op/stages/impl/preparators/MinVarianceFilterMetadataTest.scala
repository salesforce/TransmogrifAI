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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class MinVarianceFilterMetadataTest extends FlatSpec with TestSparkContext {

  val summary = MinVarianceSummary(
    dropped = Seq("f1"),
    featuresStatistics = SummaryStatistics(3, 0.01, Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3),
      Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3)),
    names = Seq("f1", "f2", "f3")
  )

  Spec[MinVarianceSummary] should "convert to and from metadata correctly" in {
    val meta = summary.toMetadata()
    meta.isInstanceOf[Metadata] shouldBe true

    val retrieved = MinVarianceSummary.fromMetadata(meta)
    retrieved.isInstanceOf[MinVarianceSummary]

    retrieved.dropped should contain theSameElementsAs summary.dropped
    retrieved.featuresStatistics.count shouldBe summary.featuresStatistics.count
    retrieved.featuresStatistics.max should contain theSameElementsAs summary.featuresStatistics.max
    retrieved.featuresStatistics.min should contain theSameElementsAs summary.featuresStatistics.min
    retrieved.featuresStatistics.mean should contain theSameElementsAs summary.featuresStatistics.mean
    retrieved.featuresStatistics.variance should contain theSameElementsAs summary.featuresStatistics.variance
    retrieved.names should contain theSameElementsAs summary.names
  }

  it should "convert to and from JSON and give the same values" in {
    val meta = summary.toMetadata()
    val json = meta.wrapped.prettyJson
    val recovered = Metadata.fromJson(json)

    // recovered shouldBe meta
    recovered.hashCode() shouldEqual summary.toMetadata().hashCode()
  }

}
