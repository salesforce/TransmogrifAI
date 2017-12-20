/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators

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
      Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3), Seq(0.1, 0.2, 0.3)),
    names = Seq("f1", "f2", "f3"),
    CategoricalStats(
      categoricalFeatures = Array("f4", "f5"),
      cramersVs = Array(0.45, 0.11),
      pointwiseMutualInfos = Map("0" -> Array(1.23, -2.11), "1" -> Array(1.11, 0.34), "2" -> Array(-0.32, 0.99)),
      mutualInfos = Array(-1.22, 0.51)
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
    retrieved.categoricalStats.cramersVs should contain theSameElementsAs summary.categoricalStats.cramersVs

    retrieved.dropped should contain theSameElementsAs summary.dropped

    retrieved.featuresStatistics.count shouldBe summary.featuresStatistics.count
    retrieved.featuresStatistics.max should contain theSameElementsAs summary.featuresStatistics.max
    retrieved.featuresStatistics.min should contain theSameElementsAs summary.featuresStatistics.min
    retrieved.featuresStatistics.mean should contain theSameElementsAs summary.featuresStatistics.mean
    retrieved.featuresStatistics.variance should contain theSameElementsAs summary.featuresStatistics.variance

    retrieved.names should contain theSameElementsAs summary.names
    retrieved.correlationsWLabel.corrType shouldBe summary.correlationsWLabel.corrType

    retrieved.categoricalStats.categoricalFeatures should contain theSameElementsAs
      summary.categoricalStats.categoricalFeatures
    retrieved.categoricalStats.cramersVs should contain theSameElementsAs summary.categoricalStats.cramersVs
  }

  it should "convert to and from JSON and give the same values" in {
    val meta = summary.toMetadata()
    val json = meta.wrapped.prettyJson
    val recovered = Metadata.fromJson(json)

    // recovered shouldBe meta
    recovered.hashCode() shouldEqual summary.toMetadata().hashCode()
  }

}
