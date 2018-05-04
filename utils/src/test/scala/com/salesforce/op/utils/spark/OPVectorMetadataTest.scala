/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.FeatureHistory
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

  type OpVectorColumnTuple = (Seq[String], Seq[String], Option[String], Option[String], Int)
  type FeatureHistoryTuple = (Seq[String], Seq[String])
  type OpVectorTuple = (String, Array[OpVectorColumnTuple], FeatureHistoryTuple)

  // AttributeGroup and Attribute require non-empty names
  val genName: Gen[String] = Gen.nonEmptyListOf(alphaNumChar).map(_.mkString)
  val genValue: Gen[String] = Gen.oneOf(genName, Gen.oneOf(Seq(OpVectorColumnMetadata.NullString)))
  val vecColTupleGen: Gen[OpVectorColumnTuple] = for {
    seq <- Gen.containerOf[Seq, String](genName)
    group <- Gen.option(genName)
    value <- Gen.option(genValue)
  } yield {
    (seq, seq, group, value, 0)
  }

  val featHistTupleGen: Gen[FeatureHistoryTuple] = Gen.zip(
    Gen.containerOf[Seq, String](genName), Gen.containerOf[Seq, String](genName)
  )
  val arrVecColTupleGen: Gen[Array[OpVectorColumnTuple]] = Gen.containerOf[Array, OpVectorColumnTuple](vecColTupleGen)

  val vecGen: Gen[OpVectorTuple] = for {
    name <- genName
    arr <- arrVecColTupleGen
    histories <- featHistTupleGen
  } yield {
    (name, arr, histories)
  }

  val seqVecGen: Gen[Seq[OpVectorTuple]] = Gen.containerOf[Seq, OpVectorTuple](vecGen)

  private def generateHistory(columnsMeta: Array[OpVectorColumnMetadata], hist: (Seq[String], Seq[String])) =
    columnsMeta.flatMap(v => v.parentFeatureName.map(p => p -> FeatureHistory(hist._1, hist._2))).toMap

  private def checkTuples(tup: OpVectorColumnTuple) = tup._1.nonEmpty && tup._2.nonEmpty


  property("column metadata stays the same when serialized to spark metadata") {
    forAll(vecColTupleGen) { (vct: OpVectorColumnTuple) =>
      if (checkTuples(vct)) {
        val columnMeta = OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5)
        columnMeta shouldEqual OpVectorColumnMetadata.fromMetadata(columnMeta.toMetadata()).head
      }
    }
  }

  property("column metadata cannot be created with empty parents or feature types") {
    forAll(vecColTupleGen) { (vct: OpVectorColumnTuple) =>
      if (!checkTuples(vct)) {
        assertThrows[AssertionError] { OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5) }
      }
    }
  }

  property("vector metadata stays the same when serialized to spark metadata") {
    forAll(vecGen) { case (outputName: String, columns: Array[OpVectorColumnTuple], hist: FeatureHistoryTuple) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)
      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history)
      val field = vectorMeta.toStructField()
      vectorMeta shouldEqual OpVectorMetadata(field)
    }
  }

  property("vector metadata properly finds indices of its columns") {
    forAll(vecGen) { case (outputName: String, columns: Array[OpVectorColumnTuple], hist: FeatureHistoryTuple) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)
      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history)
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
    forAll(seqVecGen) { (vectors: Seq[OpVectorTuple]) =>
      val vecs = vectors.map {
        case (outputName, columns, hist) =>
          val cols = columns.filter(checkTuples)
          val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
          val history = generateHistory(columnsMeta, hist)
          OpVectorMetadata(outputName, columnsMeta, history)
      }
      val flattened = OpVectorMetadata.flatten("out", vecs)
      flattened.size shouldEqual vecs.map(_.size).sum
      flattened.columns should contain theSameElementsInOrderAs vecs.flatMap(_.columns)
        .zipWithIndex.map{ case (c, i) => c.copy(index = i) }
    }
  }

  property("vector metadata should properly serialize to and from spark metadata") {
    forAll(vecGen) { case (outputName: String, columns: Array[OpVectorColumnTuple], hist: FeatureHistoryTuple) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)

      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history)

      val vectorMetaFromSerialized = OpVectorMetadata(vectorMeta.name, vectorMeta.toMetadata)
      vectorMeta.name shouldEqual vectorMetaFromSerialized.name
      vectorMeta.columns should contain theSameElementsAs vectorMetaFromSerialized.columns
      vectorMeta.history shouldEqual vectorMetaFromSerialized.history
    }
  }


  property("vector metadata should generate feature history correctly") {
    forAll(vecGen) { case (outputName: String, columns: Array[OpVectorColumnTuple], hist: FeatureHistoryTuple) =>
      val cols = columns.filter(checkTuples)
      val columnsMeta = cols.map(vct => OpVectorColumnMetadata(vct._1, vct._2, vct._3, vct._4, vct._5))
      val history = generateHistory(columnsMeta, hist)

      val vectorMeta = OpVectorMetadata(outputName, columnsMeta, history)
      if (history.isEmpty && columnsMeta.nonEmpty ) {
        assertThrows[RuntimeException](vectorMeta.getColumnHistory())
      } else {
        val colHist = vectorMeta.getColumnHistory()
        colHist.length shouldEqual columnsMeta.length
        colHist.zip(columnsMeta).foreach { case (hist, meta) =>
          hist.parentFeatureName shouldBe meta.parentFeatureName
          hist.parentFeatureType shouldBe meta.parentFeatureType
          hist.indicatorValue shouldBe meta.indicatorValue
          hist.indicatorGroup shouldBe meta.indicatorGroup
          hist.indicatorValue.contains(OpVectorColumnMetadata.NullString) shouldBe meta.isNullIndicator
          hist.parentFeatureType.foreach(p => p.contains(p) shouldBe meta.hasParentOfType(p))
          hist.parentFeatureType.exists(p => p.contains("Map") || p.contains("Prediction")) shouldBe meta.hasMapParent()
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
