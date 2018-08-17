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

import java.io.File

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.SparkMatchers._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RichDatasetTest extends FlatSpec with TestSparkContext {

  // TODO: fix implicit scope conflicts with 'org.apache.spark.sql.functions._'
  import com.salesforce.op.utils.spark.RichDataType._
  import com.salesforce.op.utils.spark.RichDataset._
  import com.salesforce.op.utils.spark.RichMetadata._

  val log = LoggerFactory.getLogger(this.getClass)

  lazy val savedPath = new File(tempDir, "richDS-" + DateTime.now().getMillis)

  private val data =
    Seq[(Integral, Real, Text, Binary, Real)](
      (Integral(1), Real(1.0), Text("abc"), Binary(false), Real(1.0)),
      (Integral(2), Real(2.0), Text("def"), Binary(true), Real(2.0)),
      (Integral(3), Real(3.0), Text.empty, Binary.empty, Real.empty)
    )

  val (ds, f1, f2, f3, f4, f5) = TestFeatureBuilder(data)

  private def field(feature: Feature[_], dataType: DataType) =
    StructField(feature.name, dataType, nullable = true)

  Spec(com.salesforce.op.utils.spark.RichDataset.getClass) should "select single features from a dataset" in {
    ds.asInstanceOf[Dataset[_]].select(f1).collect().map(_.get(0)) shouldBe Array(1, 2, 3)
    ds.select(f2).collect().map(_.get(0)) shouldBe Array(1.0, 2.0, 3.0)
    ds.select(f2).collect().map(_.get(0)) shouldBe Array(1.0, 2.0, 3.0)
    ds.select(f3).collect().map(_.get(0)) shouldBe Array("abc", "def", null)
    ds.select(f4).collect().map(_.get(0)) shouldBe Array(false, true, null)
    ds.select(f5).collect().map(_.get(0)) shouldBe Array(1.0, 2.0, null)

    ds.select(f1).schema shouldBe StructType(Array(field(f1, LongType)))
    ds.select(f2).schema shouldBe StructType(Array(field(f2, DoubleType)))
    ds.select(f3).schema shouldBe StructType(Array(field(f3, StringType)))
    ds.select(f4).schema shouldBe StructType(Array(field(f4, BooleanType)))
    ds.select(f5).schema shouldBe StructType(Array(field(f5, DoubleType)))
  }

  it should "select multiple features from a dataset" in {
    // the order of features is inverted to make sure it actually works
    val selected = ds.select(f5, f4, f3, f2, f1)

    selected.schema shouldBe StructType(Array(
      field(f5, DoubleType),
      field(f4, BooleanType),
      field(f3, StringType),
      field(f2, DoubleType),
      field(f1, LongType)
    ))

    selected.collect().map(r => (r.get(0), r.get(1), r.get(2), r.get(3), r.get(4))) shouldBe Array(
      (1.0, false, "abc", 1.0, 1),
      (2.0, true, "def", 2.0, 2),
      (null, null, null, 3.0, 3)
    )
  }

  it should "collect single features from a dataset" in {
    ds.collect(f1) shouldBe Array(1, 2, 3).map(_.toIntegral)
    ds.collect(f2) shouldBe Array(1.0, 2.0, 3.0).map(_.toReal)
    ds.collect(f3) shouldBe Array(Some("abc"), Some("def"), None).map(_.toText)
    ds.collect(f4) shouldBe Array(Some(false), Some(true), None).map(_.toBinary)
    ds.collect(f5) shouldBe Array(Some(1.0), Some(2.0), None).map(_.toReal)
  }

  it should "collect multiple features from a dataset" in {
    // the order of features is inverted to make sure it actually works
    ds.collect(f5, f4, f3, f2, f1) shouldBe Array(
      (Real(1.0), Binary(false), Text("abc"), Real(1.0), Integral(1)),
      (Real(2.0), Binary(true), Text("def"), Real(2.0), Integral(2)),
      (Real.empty, Text.empty, Text.empty, Real(3.0), Integral(3))
    )
  }

  it should "take(n) single features from a dataset" in {
    ds.take(1, f1) shouldBe Array(1.toIntegral)
    ds.take(1, f2) shouldBe Array(1.0.toReal)
    ds.take(2, f3) shouldBe Array("abc".toText, "def".toText)
    ds.take(3, f4) shouldBe Array(false.toBinary, true.toBinary, Binary.empty)
    ds.take(100, f5) shouldBe Array(1.0.toReal, 2.0.toReal, Real.empty)
  }

  it should "take(n) multiple features from a dataset" in {
    // the order of features is inverted to make sure it actually works
    ds.take(2, f5, f4, f3, f2, f1) shouldBe Array(
      (1.0.toReal, false.toBinary, "abc".toText, 1.0.toReal, 1.toIntegral),
      (2.0.toReal, true.toBinary, "def".toText, 2.0.toReal, 2.toIntegral)
    )
  }

  it should "get metadata for features from a dataset" in {
    val metadata = new MetadataBuilder().putString("foo", "bar").build()
    val dsWithMeta = ds.select(col(f1.name).as(f1.name, metadata), col(f2.name).as(f2.name))
    dsWithMeta.metadata(f1, f2) shouldBe Map(f1 -> metadata, f2 -> Metadata.empty)
  }

  it should "be empty when isEmpty is true and vice versa" in {
    ds.count().toInt should be > 0
    ds.isEmpty shouldBe false

    val empty = ds.filter(_ => false)
    empty.count() shouldBe 0
    empty.isEmpty shouldBe true
  }

  it should "throw an error if a dataset schema does not match the features" in {
    the[IllegalArgumentException] thrownBy ds.select(f1.copy(name = "blarg"))
    the[IllegalArgumentException] thrownBy ds.select(f2.copy(name = f1.name))
    the[IllegalArgumentException] thrownBy ds.collect(f1.copy(name = "blarg"))
    the[IllegalArgumentException] thrownBy ds.collect(f2.copy(name = f1.name))
    the[IllegalArgumentException] thrownBy ds.metadata(f1.copy(name = "blarg"))
    the[IllegalArgumentException] thrownBy ds.metadata(f2.copy(name = f1.name))
  }

  it should "save & load a dataset and schema" in {
    val meta = new MetadataBuilder().putString("foo", "bar").build()
    val vect = udf((v: Long) => if (v % 2 == 0) Vectors.dense(Array(v.toDouble)) else null)
    val transformed =
      ds.select(col("*"), col(f1.name).cast(StringType).as("f1_(str)", meta))
        .withColumn("vector", vect(col(f1.name)))

    transformed.saveAvro(savedPath.toString)
    val loaded = loadAvro(savedPath.toString)

    assertDataFrames(actual = loaded, expected = transformed)
  }

  it should "check 'forall' with names" in {
    ds.forall[Long]("f1")(i => i > 0) shouldBe true
    ds.forall[Double]("f2")(0.0 ==) shouldBe false
  }

  it should "check 'exists' with names" in {
    ds.exists[Long]("f1")(2 ==) shouldBe true
    ds.exists[Double]("f2")(3.5 <) shouldBe false
    ds.exists[String]("f3")(s => s == null || (s startsWith "d")) shouldBe true
    ds.exists[String]("f3")(Option(_) forall (_ contains 'e')) shouldBe true
  }

  it should "check 'forNone' with names" in {
    ds.forNone[Long]("f1")(42 ==) shouldBe true
    ds.forNone[Double]("f2")(1.5 <) shouldBe false
    ds.forNone[String]("f3")(s => s != null && s(0) == 'z') shouldBe true
    ds.forNone[String]("f3")(Option(_) exists (_ contains 'x')) shouldBe true
  }

  it should "check 'forall' with features" in {
    ds.forall(f1)(_ exists (_ > 0)) shouldBe true
    ds.forall(f2)(Some(0.0) ==) shouldBe false
  }

  it should "check 'exists' with features" in {
    ds.exists(f1)(Some(2) ==) shouldBe true
    ds.exists(f2)(_ exists (3.5 <)) shouldBe false
    ds.exists(f3)(_ exists(_ startsWith "d")) shouldBe true
    ds.exists(f3)(_ forall (_ contains 'e')) shouldBe true
  }

  it should "check 'forNone' with features" in {
    ds.forNone(f1)(Some(42) ==) shouldBe true
    ds.forNone(f2)(_ exists (1.5 <)) shouldBe false
    ds.forNone(f3)(_ exists(_ startsWith "z")) shouldBe true
    ds.forNone(f3)(_ exists (_ contains 'x')) shouldBe true
  }

  it should "check 'forall' via SparkMatchers" in {
    in(ds) allOf "f1" should ((i: Long) => i > 0)
  }

  it should "check 'exists' via SparkMatchers" in {
    in(ds) someOf "f1" shouldBe 2
    in(ds) someOf "f3" should ((s: String) => s == null || (s startsWith "d"))
    in(ds) someOf "f3" should ((x: String) => Option(x) forall (_ contains 'e'))
  }

  it should "check 'forNone' via SparkMatchers" in {
    1 should be < 3
    in(ds) noneOf "f1" shouldBe 42
    in(ds) noneOf "f3" should ((s: String) => s != null && s(0) == 'z')
    in(ds) noneOf "f3"should ((x: String) => Option(x) exists (_ contains 'x'))
    in(ds) noneOf "f2" shouldBe 0.0
    in(ds) noneOf "f2" should ((x: Double) => x > 3.5)
  }

  it should "only compare types and ignore nullability" in {
    val arrayNullType = StructType(Array(field(f1, ArrayType(StringType, containsNull = false))))
    val arrayNonNullableType = StructType(Array(field(f1, ArrayType(StringType, containsNull = true))))
    arrayNullType.equalsIgnoreNullability(arrayNonNullableType) shouldBe true

    val mapNullType = StructType(Array(field(f1, MapType(StringType, StringType, valueContainsNull = false))))
    val mapNonNullableType = StructType(Array(field(f1, MapType(StringType, StringType, valueContainsNull = true))))
    mapNullType.equalsIgnoreNullability(mapNonNullableType) shouldBe true
  }

  private def assertDataFrames(actual: DataFrame, expected: DataFrame): Unit = {
    if (log.isInfoEnabled) {
      log.info("Actual schema:\n" + actual.schema.prettyJson)
      actual.show(false)
      log.info("Expected schema:\n" + expected.schema.prettyJson)
      expected.show(false)
    }
    // assert the columns
    actual.columns shouldBe expected.columns
    // assert metadata
    for {
      (l, t) <- actual.schema.fields.zip(expected.schema.fields)
      _ = l.metadata.underlyingMap.size should be >= t.metadata.underlyingMap.size
      (k, v) <- t.metadata.underlyingMap
    } {
      l.metadata.contains(k) shouldBe true
      l.metadata.underlyingMap(k) shouldBe v
    }
    // assert schema ignoring nullability (for now)
    actual.schema.equalsIgnoreNullability(expected.schema) shouldBe true
    // assert the data
    actual.collect() should contain theSameElementsAs expected.collect()
  }

}
