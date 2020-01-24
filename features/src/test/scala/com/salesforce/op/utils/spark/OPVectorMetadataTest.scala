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

package com.salesforce.op.utils.spark

import com.salesforce.op.{FeatureHistory, SensitiveFeatureInformation}
import com.salesforce.op.features.types.{DateTime, Email, FeatureType, OPMap, PickList, Prediction, Real, RealMap, TextAreaMap}
import com.salesforce.op.test.TestCommon
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalacheck.Gen.alphaNumChar
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks


@RunWith(classOf[JUnitRunner])
class OPVectorMetadataTest extends PropSpec with TestCommon with PropertyChecks {

  type OpVectorColumnTuple = (Seq[String], Seq[String], Option[String], Option[String], Option[String], Int)
  type FeatureHistoryTuple = (Seq[String], Seq[String])

  type SensitiveTuple = (SensitiveNameTuple, String, Option[String], Boolean)
  type SensitiveNameTuple = (Double, Seq[String], Double, Double, Double)

  type OpVectorTuple = (String, Array[OpVectorColumnTuple], FeatureHistoryTuple, Seq[SensitiveTuple])

  // AttributeGroup and Attribute require non-empty names
  val genName: Gen[String] = Gen.nonEmptyListOf(alphaNumChar).map(_.mkString)
  val genType: Gen[String] = Gen.oneOf(
    FeatureType.typeName[DateTime], FeatureType.typeName[Email], FeatureType.typeName[PickList],
    FeatureType.typeName[Prediction], FeatureType.typeName[Real], FeatureType.typeName[RealMap],
    FeatureType.typeName[TextAreaMap]
  )
  val genValue: Gen[String] = Gen.oneOf(genName, Gen.oneOf(Seq(OpVectorColumnMetadata.NullString)))
  val vecColTupleGen: Gen[OpVectorColumnTuple] = for {
    nameSeq <- Gen.containerOf[Seq, String](genName)
    typeSeq <- Gen.listOfN(nameSeq.length, genType)
    group <- Gen.option(genName)
    ivalue <- Gen.option(genValue)
    dvalue <- Gen.option(genValue)
  } yield {
    (nameSeq, typeSeq, group, ivalue, if (ivalue.isEmpty) dvalue else None, 0)
  }

  val featHistTupleGen: Gen[FeatureHistoryTuple] = Gen.zip(
    Gen.containerOf[Seq, String](genName), Gen.containerOf[Seq, String](genName)
  )
  val arrVecColTupleGen: Gen[Array[OpVectorColumnTuple]] = Gen.containerOf[Array, OpVectorColumnTuple](vecColTupleGen)

  val sensitiveGen: Gen[SensitiveTuple] = for {
    featureName <- genName
    mapKey <- Gen.option(genName)
    actionTaken <- Gen.oneOf[Boolean](Seq(false, true))
    probName <- Gen.choose(0.0, 1.0)
    genderDetectResults <- Gen.containerOf[Seq, String](genName)
    probMale <- Gen.choose(0.0, 1.0)
    probFemale <- Gen.choose(0.0, 1.0 - probMale)
    probOther <- Gen.choose(0.0, 1.0 - probMale - probFemale)
  } yield {
    ((probName, genderDetectResults, probMale, probFemale, probOther), featureName, mapKey, actionTaken)
  }

  val vecGen: Gen[OpVectorTuple] = for {
    name <- genName
    arr <- arrVecColTupleGen
    histories <- featHistTupleGen
    sensitiveCols <- Gen.containerOf[Seq, SensitiveTuple](sensitiveGen)
  } yield {
    (name, arr, histories, sensitiveCols)
  }

  val seqVecGen: Gen[Seq[OpVectorTuple]] = Gen.containerOf[Seq, OpVectorTuple](vecGen)

  private def generateHistory(
    columnsMeta: Array[OpVectorColumnMetadata], hist: (Seq[String], Seq[String])
  ): Map[String, FeatureHistory] =
    columnsMeta.flatMap(v => v.parentFeatureName.map(p => p -> FeatureHistory(hist._1, hist._2))).toMap

  private def generateSensitiveFeatureInfo(
    columnsMeta: Array[OpVectorColumnMetadata], sensitiveInfoSeqRaw: Seq[SensitiveTuple]
  ): Map[String, Seq[SensitiveFeatureInformation]] = {
    val sensitiveInfoSeq = sensitiveInfoSeqRaw map {
      case ((probName, genderDetectResults, probMale, probFemale, probOther), featureName, mapKey, actionTaken) =>
        SensitiveNameInformation(
          probName, genderDetectResults, probMale, probFemale, probOther, featureName, mapKey, actionTaken
        )
    }
    columnsMeta.flatMap(v => v.parentFeatureName.map(p => p -> sensitiveInfoSeq)).toMap
  }

  private def checkTuples(tup: OpVectorColumnTuple): Boolean = tup._1.nonEmpty && tup._2.nonEmpty

  property("column metadata stays the same when serialized to spark metadata") {
    forAll(vecColTupleGen) { vct: OpVectorColumnTuple =>
      if (checkTuples(vct)) {
        val columnMeta = OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5)
        columnMeta shouldEqual OpVectorColumnMetadata.fromMetadata(columnMeta.toMetadata()).head
      }
    }
  }

  property("column metadata cannot be created with empty parents or feature types") {
    forAll(vecColTupleGen) { vct: OpVectorColumnTuple =>
      if (!checkTuples(vct)) {
        assertThrows[IllegalArgumentException] { OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5) }
      }
    }
  }

  property("vector metadata stays the same when serialized to spark metadata") {
    forAll(vecGen) {
      case (outputName: String,
        columns: Array[OpVectorColumnTuple],
        hist: FeatureHistoryTuple,
        sens: Seq[SensitiveTuple]
      ) if outputName.nonEmpty =>
        val cols = columns.filter(checkTuples)
        val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
        val history = generateHistory(columnsMeta, hist)
        val sensitive = generateSensitiveFeatureInfo(columnsMeta, sens)
        val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history, sensitive)
        val field = vectorMeta.toStructField()
        vectorMeta shouldEqual OpVectorMetadata(field)
      case _ => true shouldEqual true
    }
  }

  property("vector metadata properly finds indices of its columns") {
    forAll(vecGen) {
      case (outputName: String,
        columns: Array[OpVectorColumnTuple],
        hist: FeatureHistoryTuple,
        sens: Seq[SensitiveTuple]) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)
      val sensitive = generateSensitiveFeatureInfo(columnsMeta, sens)
      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history, sensitive)
      for {(col, i) <- vectorMeta.columns.zipWithIndex} {
        vectorMeta.index(col) shouldEqual i
      }
    }
  }

  val emptyMetadata = Table("meta", Metadata.empty)

  property("vector metadata throws exception on empty metadata") {
    forAll(emptyMetadata) { empty =>
      assertThrows[RuntimeException] {
        OpVectorMetadata("field", empty)
      }
    }
  }

  property("vector metadata flattens correctly") {
    forAll(seqVecGen) { vectors: Seq[OpVectorTuple] =>
      val vecs = vectors.map {
        case (outputName, columns, hist, sens) =>
          val cols = columns.filter(checkTuples)
          val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
          val history = generateHistory(columnsMeta, hist)
          val sensitive = generateSensitiveFeatureInfo(columnsMeta, sens)
          OpVectorMetadata(outputName, columnsMeta, history, sensitive)
      }
      val flattened = OpVectorMetadata.flatten("out", vecs)
      flattened.size shouldEqual vecs.map(_.size).sum
      flattened.columns should contain theSameElementsInOrderAs vecs.flatMap(_.columns)
        .zipWithIndex.map{ case (c, i) => c.copy(index = i) }
    }
  }

  property("vector metadata should properly serialize to and from spark metadata") {
    forAll(vecGen) {
      case (outputName: String,
        columns: Array[OpVectorColumnTuple],
        hist: FeatureHistoryTuple,
        sens: Seq[SensitiveTuple]) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)
      val sensitive = generateSensitiveFeatureInfo(columnsMeta, sens)
      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history, sensitive)

      val vectorMetaFromSerialized = OpVectorMetadata(vectorMeta.name, vectorMeta.toMetadata)
      vectorMeta.name shouldEqual vectorMetaFromSerialized.name
      vectorMeta.columns should contain theSameElementsAs vectorMetaFromSerialized.columns
      vectorMeta.history shouldEqual vectorMetaFromSerialized.history
    }
  }


  property("vector metadata should generate feature history correctly") {
    forAll(vecGen) { case (
      outputName: String,
      columns: Array[OpVectorColumnTuple],
      hist: FeatureHistoryTuple,
      sens: Seq[SensitiveTuple]) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)
      val sensitive = generateSensitiveFeatureInfo(columnsMeta, sens)
      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history, sensitive)

      if (history.isEmpty && columnsMeta.nonEmpty) {
        assertThrows[RuntimeException](vectorMeta.getColumnHistory())
      } else {
        val colHist = vectorMeta.getColumnHistory()
        colHist.length shouldEqual columnsMeta.length
        colHist.zip(columnsMeta).foreach { case (hist, meta) =>
          hist.parentFeatureName shouldBe meta.parentFeatureName
          hist.parentFeatureType shouldBe meta.parentFeatureType
          hist.indicatorValue shouldBe meta.indicatorValue
          hist.grouping shouldBe meta.grouping
          hist.descriptorValue shouldBe meta.descriptorValue
          hist.indicatorValue.contains(OpVectorColumnMetadata.NullString) shouldBe meta.isNullIndicator
          hist.parentFeatureType.exists(p => p.contains("Map") || p.contains("Prediction")) shouldBe
            meta.hasParentOfSubType[OPMap[_]]
        }
        if (colHist.nonEmpty && colHist.head.parentFeatureName.nonEmpty) {
          colHist.head.parentFeatureName.flatMap(p => history(p).stages).distinct.sorted should
            contain theSameElementsAs colHist.head.parentFeatureStages
          colHist.head.parentFeatureName.flatMap(p => history(p).originFeatures).distinct.sorted should
            contain theSameElementsAs colHist.head.parentFeatureOrigins
        }
      }
    }
  }
}
