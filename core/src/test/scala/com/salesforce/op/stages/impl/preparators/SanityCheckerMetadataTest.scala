/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.preparators

import scala.util.Try
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SanityCheckerMetadataTest extends FlatSpec with TestSparkContext {

  val summary = SanityCheckerSummary(
    correlationsWLabel = Correlations(Seq("f2", "f3"), Seq(0.2, 0.3), Seq(), CorrelationType.Pearson),
    dropped = Seq("f1"),
    featuresStatistics = SummaryStatistics(3, 0.01, Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3),
      Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3)),
    names = Seq("f1", "f2", "f3"),
    Array(
      CategoricalGroupStats(
        group = "f4",
        categoricalFeatures = Array("f4"),
        contingencyMatrix = Map("0" -> Array(12), "1" -> Array(12), "2" -> Array(12)),
        cramersV = 0.45,
        pointwiseMutualInfo = Map("0" -> Array(1.23), "1" -> Array(1.11), "2" -> Array(-0.32)),
        mutualInfo = -1.22,
        maxRuleConfidences = Array(1.0),
        supports = Array(1.0)
      ),
      CategoricalGroupStats(
        group = "f5",
        categoricalFeatures = Array("f5"),
        contingencyMatrix = Map("0" -> Array(12), "1" -> Array(12), "2" -> Array(12)),
        cramersV = 0.11,
        pointwiseMutualInfo = Map("0" -> Array(-2.11), "1" -> Array(0.34), "2" -> Array(0.99)),
        mutualInfo = -0.51,
        maxRuleConfidences = Array(1.0),
        supports = Array(1.0)
      )
    )
  )

  Spec[SanityCheckerSummary] should "convert to and from metadata correctly" in {
    val meta = summary.toMetadata()

    meta.isInstanceOf[Metadata] shouldBe true

    val retrieved = SanityCheckerSummary.fromMetadata(meta)
    retrieved.isInstanceOf[SanityCheckerSummary]
    retrieved.correlationsWLabel.nanCorrs should contain theSameElementsAs summary.correlationsWLabel.nanCorrs

    retrieved.correlationsWLabel.featuresIn should contain theSameElementsAs summary.correlationsWLabel.featuresIn
    retrieved.correlationsWLabel.values should contain theSameElementsAs summary.correlationsWLabel.values
    retrieved.categoricalStats.map(_.cramersV) should contain theSameElementsAs
      summary.categoricalStats.map(_.cramersV)

    retrieved.dropped should contain theSameElementsAs summary.dropped

    retrieved.featuresStatistics.count shouldBe summary.featuresStatistics.count
    retrieved.featuresStatistics.max should contain theSameElementsAs summary.featuresStatistics.max
    retrieved.featuresStatistics.min should contain theSameElementsAs summary.featuresStatistics.min
    retrieved.featuresStatistics.mean should contain theSameElementsAs summary.featuresStatistics.mean
    retrieved.featuresStatistics.variance should contain theSameElementsAs summary.featuresStatistics.variance

    retrieved.names should contain theSameElementsAs summary.names
    retrieved.correlationsWLabel.corrType shouldBe summary.correlationsWLabel.corrType

    retrieved.categoricalStats.flatMap(_.categoricalFeatures) should contain theSameElementsAs
      summary.categoricalStats.flatMap(_.categoricalFeatures)
    retrieved.categoricalStats.map(_.cramersV) should contain theSameElementsAs
      summary.categoricalStats.map(_.cramersV)
  }

  it should "convert to and from JSON and give the same values" in {
    val meta = summary.toMetadata()
    val json = meta.wrapped.prettyJson
    val recovered = Metadata.fromJson(json)

    // recovered shouldBe meta
    recovered.hashCode() shouldEqual summary.toMetadata().hashCode()
  }

  it should "be able to be constructed (albeit poorly) from the old metadata format" in {
    val oldMeta = Metadata.fromJson(loadResource("/SanityCheckerSummaryMetadata_pre3.3.0.json"))
    Try(SanityCheckerSummary.fromMetadata(oldMeta)).isSuccess shouldBe true
  }

}
