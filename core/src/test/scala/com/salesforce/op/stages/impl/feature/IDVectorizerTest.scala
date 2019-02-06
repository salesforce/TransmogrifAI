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

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class IDVectorizerTest extends OpEstimatorSpec[OPVector, SequenceModel[ID, OPVector], IDVectorizer] {

  lazy val (inputData, f1, f2) = TestFeatureBuilder("id1", "id2",
    Seq[(ID, ID)](
      ("00QBC72".toID, "00QBC.72".toID),
      ("00QBC72".toID, "0001".toID),
      ("ZX72".toID, "1".toID),
      ("00QBC72".toID, "GM-300-44".toID),
      (ID.empty, ID.empty)
    )
  )
  val estimator = new IDVectorizer()
    .setMaxCardinality(2).setMinSupport(1)
    .setTopK(2)
    .setInput(f1, f2)


  val expectedResult = Seq(
    Vectors.sparse(4, Array(0), Array(1.0)),
    Vectors.sparse(4, Array(0), Array(1.0)),
    Vectors.sparse(4, Array(1), Array(1.0)),
    Vectors.sparse(4, Array(0), Array(1.0)),
    Vectors.sparse(4, Array(3), Array(1.0))
  ).map(_.toOPVector)

  it should "detect one categorical and one non-categorical id feature" in {
    val dropIDed = new IDVectorizer()
      .setMaxCardinality(2)
      .setMinSupport(1)
      .setInput(f1, f2).getOutput()


    val transformed = new OpWorkflow()
      .setResultFeatures(dropIDed).setInputDataset(inputData).train().score()
    val result = transformed.collect(dropIDed)
    val field = transformed.schema(dropIDed.name)
    val expected = Seq(
      Vectors.sparse(4, Array(0), Array(1.0)),
      Vectors.sparse(4, Array(0), Array(1.0)),
      Vectors.sparse(4, Array(1), Array(1.0)),
      Vectors.sparse(4, Array(0), Array(1.0)),
      Vectors.sparse(4, Array(3), Array(1.0))
    ).map(_.toOPVector)


    val meta = OpVectorMetadata(field)

    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 4
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
    result shouldBe expected
  }

  it should "detect two categorical id features" in {
    val dropIDed = new IDVectorizer()
      .setMaxCardinality(10).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()


    val transformed = new OpWorkflow().setResultFeatures(dropIDed).setInputDataset(inputData).train().score()
    val field = transformed.schema(dropIDed.name)
    val result = transformed.collect(dropIDed)

    val expected = Seq(
      Vectors.sparse(8, Array(0, 5), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 4), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 6), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(3, 7), Array(1.0, 1.0))
    ).map(_.toOPVector)

    result shouldBe expected

    val meta = OpVectorMetadata(field)
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 8
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
      } else if (col.index == 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }

  it should "detect two non categorical id features" in {
    val dropIDed = new IDVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow()
      .setResultFeatures(dropIDed).setInputDataset(inputData).train().score()
    val result = transformed.collect(dropIDed)
    val expected = Seq(
      OPVector.empty,
      OPVector.empty,
      OPVector.empty,
      OPVector.empty,
      OPVector.empty
    )

    result shouldBe expected

  }


  it should "fail with an error" in {
    val emptyDF = inputData.filter(inputData("id1") === "").toDF()

    val dropIDed = new IDVectorizer()
      .setMaxCardinality(2).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val thrown = intercept[IllegalArgumentException] {
      new OpWorkflow().setResultFeatures(dropIDed).setInputDataset(emptyDF).train().score()
    }
    assert(thrown.getMessage.contains("requirement failed"))
  }

  it should "combined dropped and non dropped" in {
    val dropped = new IDVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val kept = new IDVectorizer()
      .setMaxCardinality(4).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val combined = Seq(dropped, kept, dropped).combine()

    val transformed = new OpWorkflow().setResultFeatures(combined).setInputDataset(inputData).train().score()
    val field = transformed.schema(combined.name)
    val result = transformed.collect(combined)

    val expected = Seq(
      Vectors.sparse(8, Array(0, 5), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 4), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 6), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(3, 7), Array(1.0, 1.0))
    ).map(_.toOPVector)

    result shouldBe expected

    val meta = OpVectorMetadata(field)
    meta.history.keys shouldBe Set(f1.name, f2.name)
    meta.columns.length shouldBe 8
    meta.columns.foreach { col =>
      if (col.index < 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
      } else if (col.index == 2) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.OtherString)
      } else if (col.index == 3) {
        col.parentFeatureName shouldBe Seq(f1.name)
        col.grouping shouldBe Option(f1.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      } else if (col.index < 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
      } else if (col.index == 6) {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.OtherString)
      } else {
        col.parentFeatureName shouldBe Seq(f2.name)
        col.grouping shouldBe Option(f2.name)
        col.indicatorValue shouldBe Option(OpVectorColumnMetadata.NullString)
      }
    }
  }
}
