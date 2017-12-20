/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType._
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestOpVectorMetadataBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import com.salesforce.op.{OpWorkflow, _}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TransmogrifyTest extends FlatSpec with PassengerSparkFixtureTest {

  val inputFeatures = Seq(heightNoWindow, weight, gender)

  Spec(Transmogrifier.getClass) should "return a single output feature of type vector with the correct name" in {
    val feature = inputFeatures.transmogrify()
    feature.name.contains("gender-heightNoWindow-weight_3-stagesApplied_OPVector")
  }

  it should "return a model when fitted" in {
    val feature = inputFeatures.transmogrify()
    val model = new OpWorkflow().setResultFeatures(feature).setReader(dataReader).train()

    model.getResultFeatures() should contain theSameElementsAs Array(feature)
    val name = model.getResultFeatures().map(_.name).head
    name.contains("gender-heightNoWindow-weight_3-stagesApplied_OPVector")
  }

  it should "correctly transform the data and store the feature names in metadata" in {
    val feature = inputFeatures.transmogrify()
    val model = new OpWorkflow().setResultFeatures(feature).setReader(dataReader).train()
    val transformed = model.score(keepRawFeatures = true, keepIntermediateFeatures = true)
    val hist = feature.parents.flatMap{ f =>
      val h = f.history()
      h.originFeatures.map(o => o -> FeatureHistory(Seq(o), h.stages))
    }.toMap

    transformed.schema.toOpVectorMetadata(feature.name) shouldEqual
      TestOpVectorMetadataBuilder.withOpNamesAndHist(
        feature.originStage,
        hist,
        (gender, "vecSet", List(IndCol(Some("OTHER")), IndCol(Some(TransmogrifierDefaults.NullString)))),
        (heightNoWindow, "vecReal", List(RootCol,
          IndColWithGroup(Some(TransmogrifierDefaults.NullString), heightNoWindow.name))),
        (weight, "vecReal", List(RootCol, IndColWithGroup(Some(TransmogrifierDefaults.NullString), weight.name)))
      )

    transformed.schema.findFields("heightNoWindow-weight_1-stagesApplied_OPVector").nonEmpty shouldBe true

    val collected = transformed.collect(feature)

    collected.head.v.size shouldEqual 6
    collected.map(_.v.toArray.toList).toSet shouldEqual
      Set(
        List(0.0, 1.0, 211.4, 1.0, 96.0, 1.0),
        List(1.0, 0.0, 172.0, 0.0, 78.0, 0.0),
        List(1.0, 0.0, 168.0, 0.0, 67.0, 0.0),
        List(1.0, 0.0, 363.0, 0.0, 172.0, 0.0),
        List(1.0, 0.0, 186.0, 0.0, 96.0, 0.0)
      )
  }

}
