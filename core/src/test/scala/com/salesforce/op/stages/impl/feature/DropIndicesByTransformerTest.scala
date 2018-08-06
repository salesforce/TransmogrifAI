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

import com.salesforce.op._
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.apache.spark.sql.functions._



@RunWith(classOf[JUnitRunner])
class DropIndicesByTransformerTest extends OpTransformerSpec[OPVector, DropIndicesByTransformer] {

  val (inputData, transformer) = {
    val vecData = Seq(
      Vectors.dense(1.0, 1.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0),
      Vectors.dense(0.0)
    ).map(_.toOPVector)
    val (data, v) = TestFeatureBuilder(vecData)
    val meta = OpVectorMetadata(v.name, Array(TransientFeature(v).toColumnMetaData()), Map.empty).toMetadata
    val inputData = data.withColumn(v.name, col(v.name).as(v.name, meta))
    val stage = new DropIndicesByTransformer(_.isNullIndicator).setInput(v).setInputSchema(inputData.schema)
    inputData -> stage
  }

  val expectedResult = Seq(
    Vectors.dense(1.0), Vectors.dense(0.0), Vectors.dense(0.0)
  ).map(_.toOPVector)

  val picklistData = RandomText.pickLists(domain = List("Red", "Blue", "Green")).withProbabilityOfEmpty(0.0).limit(100)
  val (df, picklistFeature) = TestFeatureBuilder("color", picklistData)

  it should "filter vector using a predicate" in {
    val vectorizedPicklist = picklistFeature.vectorize(topK = 10, minSupport = 3, cleanText = false)
    val prunedVector = new DropIndicesByTransformer(_.indicatorValue.contains("Red"))
      .setInput(vectorizedPicklist)
      .getOutput()
    val materializedFeatures = new OpWorkflow().setResultFeatures(vectorizedPicklist, prunedVector).transform(df)

    materializedFeatures.collect(prunedVector).foreach(_.value.size shouldBe 4)
    materializedFeatures.collect().foreach { r =>
      if (r.getString(0) == "Red") r.getAs[Vector](2).toArray.forall(_ == 0) shouldBe true
      else r.getAs[Vector](2).toArray.max shouldBe 1
    }
    val rawMeta = OpVectorMetadata(vectorizedPicklist.name, vectorizedPicklist.originStage.getMetadata())
    val trimmedMeta = OpVectorMetadata(materializedFeatures.schema(prunedVector.name))
    rawMeta.columns.length - 1 shouldBe trimmedMeta.columns.length
    trimmedMeta.columns.foreach(_.indicatorValue == "Red" shouldBe false)
  }

  it should "work with its shortcut" in {
    val vectorizedPicklist = picklistFeature.vectorize(topK = 10, minSupport = 3, cleanText = false)
    val prunedVector = vectorizedPicklist.dropIndicesBy(_.isNullIndicator)
    val materializedFeatures = new OpWorkflow().setResultFeatures(vectorizedPicklist, prunedVector).transform(df)

    materializedFeatures.collect(prunedVector).foreach(_.value.size shouldBe 4)
    materializedFeatures.collect().foreach( _.getAs[Vector](2).toArray.max shouldBe 1)
    val rawMeta = OpVectorMetadata(vectorizedPicklist.name, vectorizedPicklist.originStage.getMetadata())
    val trimmedMeta = OpVectorMetadata(materializedFeatures.schema(prunedVector.name))
    rawMeta.columns.length - 1 shouldBe trimmedMeta.columns.length
    trimmedMeta.columns.foreach(_.isNullIndicator shouldBe false)
  }

  it should "validate that the match function is serializable" in {
    class NonSerializable(val in: Int)
    val nonSer = new NonSerializable(5)
    val vectorizedPicklist = picklistFeature.vectorize(topK = 10, minSupport = 3, cleanText = false)
    intercept[IllegalArgumentException](
      vectorizedPicklist.dropIndicesBy(_.indicatorValue.get == nonSer.in)
    ).getMessage shouldBe "Provided function is not serializable"
  }

}
