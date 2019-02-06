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
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.spark.RichDataset._

@RunWith(classOf[JUnitRunner])
class IDMapVectorizerTest  extends FlatSpec with TestSparkContext {
  lazy val (data, m1, m2, f1, f2) = TestFeatureBuilder("idMap1", "idMap2", "id1", "id2",
    Seq[(IDMap, IDMap, ID, ID)](
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "0345963CD.")), IDMap.empty,
        "0345963CD".toID, "0345963CD.".toID),
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "15")), IDMap.empty,
        "0345963CD".toID, "15".toID),
      (IDMap(Map("id1" -> "0000EAS", "id2" -> "ISSS5-3f0")), IDMap.empty,
        "0000EAS".toID, "ISSS5-3f0".toID),
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "40554-SSSe")), IDMap.empty,
        "0345963CD".toID, "40554-SSSe".toID),
      (IDMap.empty, IDMap.empty, ID.empty, ID.empty)
    )
  )

  lazy val (data2, m3, m4, f3, f4) = TestFeatureBuilder("idMap1", "idMap2", "id1", "id2",
    Seq[(IDMap, IDMap, Text, Text)](
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "0345963CD!")), IDMap.empty,
        "0345963CD".toID, "0345963CD!".toID),
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "15")), IDMap.empty,
        "0345963CD".toID, "15".toID),
      (IDMap(Map("id1" -> "0000EAS", "id2" -> "ISSS5-3f0")), IDMap.empty,
        "0000EAS".toID, "ISSS5-3f0".toID),
      (IDMap(Map("id1" -> "0345963CD", "id2" -> "40554-SSSe")), IDMap.empty,
        "0345963CD".toID, "40554-SSSe".toID),
      (IDMap.empty, IDMap.empty, TextArea.empty, TextArea.empty)
    )
  )



  Spec[IDMapVectorizer] should "detect one categorical and one non-categorical text feature" in {
    val idMapVectorized = new IDMapVectorizer()
      .setMaxCardinality(2).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorized = new IDVectorizer()
      .setMaxCardinality(2).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(idMapVectorized, idVectorized).setInputDataset(data)
      .train().score()
    val result = transformed.collect(idMapVectorized, idVectorized)
    val mapMeta = OpVectorMetadata(transformed.schema(idMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(idVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index < 4) m.grouping shouldBe f.grouping
      else m.grouping shouldBe Option(f2.name)
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "detect two categorical text features" in {
    val idMapVectorized = new IDMapVectorizer()
      .setMaxCardinality(10).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorized = new IDVectorizer()
      .setMaxCardinality(10).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(idMapVectorized, idVectorized).setInputDataset(data)
      .train().score()
    val result = transformed.collect(idMapVectorized, idVectorized)
    val mapMeta = OpVectorMetadata(transformed.schema(idMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(idVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      m.grouping shouldBe f.grouping
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "drop all long IDs" in {
    val idMapVectorized = new IDMapVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorized = new IDVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(idMapVectorized, idVectorized).setInputDataset(data)
      .train().score()
    val result = transformed.collect(idMapVectorized, idVectorized)
    val mapMeta = OpVectorMetadata(transformed.schema(idMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(idVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns shouldBe Array.empty

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "combine dropped and kept" in {
    val idMapVectorized = new IDMapVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorized = new IDVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()


    val idMapVectorizedKept = new IDMapVectorizer()
      .setMaxCardinality(2).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorizedKept = new IDVectorizer()
      .setMaxCardinality(2).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()


    val combined = Seq(idVectorizedKept, idVectorized).combine()
    val combinedMap = Seq(idMapVectorized, idMapVectorizedKept).combine()

    val transformed = new OpWorkflow().setResultFeatures(combined, combinedMap).setInputDataset(data)
      .train().score()
    val result = transformed.collect(combined, combinedMap)
    val mapMeta = OpVectorMetadata(transformed.schema(combinedMap.name))
    val meta = OpVectorMetadata(transformed.schema(combined.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      m.grouping shouldBe f.grouping
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

  it should "use shared hash space for two text features again" in {
    val idMapVectorized = new IDMapVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setCleanKeys(false)
      .setInput(m1, m2).getOutput()

    val idVectorized = new IDVectorizer()
      .setMaxCardinality(1).setMinSupport(1).setTopK(2)
      .setInput(f1, f2).getOutput()

    val transformed = new OpWorkflow().setResultFeatures(idMapVectorized, idVectorized).setInputDataset(data)
      .train().score()
    val result = transformed.collect(idMapVectorized, idVectorized)
    val mapMeta = OpVectorMetadata(transformed.schema(idMapVectorized.name))
    val meta = OpVectorMetadata(transformed.schema(idVectorized.name))
    mapMeta.history.keys shouldBe Set(m1.name, m2.name)
    mapMeta.columns.length shouldBe meta.columns.length

    mapMeta.columns.zip(meta.columns).foreach { case (m, f) =>
      m.parentFeatureName shouldBe Array(m1.name)
      m.parentFeatureType shouldBe Array(m1.typeName)
      if (m.index == 10) {
        assert(m.grouping === Option(f1.name), s"first null indicator should be from ${f1.name}")
      } else if (m.index > 10) {
        assert(m.grouping === Option(f2.name), s"second null indicator should be from ${f2.name}")
      }
      m.indicatorValue shouldBe f.indicatorValue
    }

    result.foreach { case (vec1, vec2) => vec1 shouldBe vec2 }
  }

}
