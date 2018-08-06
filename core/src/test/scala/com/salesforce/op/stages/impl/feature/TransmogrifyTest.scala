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

import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestOpVectorColumnType._
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestOpVectorMetadataBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import com.salesforce.op._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TransmogrifyTest extends FlatSpec with PassengerSparkFixtureTest {

  val inputFeatures = Array[OPFeature](heightNoWindow, weight, gender)

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
    val feature = inputFeatures.toSeq.transmogrify()
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
