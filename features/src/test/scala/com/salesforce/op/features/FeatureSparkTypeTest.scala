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

package com.salesforce.op.features

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.reflect.runtime.universe._

@RunWith(classOf[JUnitRunner])
class FeatureSparkTypeTest extends FlatSpec with TestSparkContext {
  val sparkTypeToTypeTagMappings = Seq(
    (DoubleType, weakTypeTag[types.RealNN]), (FloatType, weakTypeTag[types.RealNN]),
    (LongType, weakTypeTag[types.Integral]), (IntegerType, weakTypeTag[types.Integral]),
    (ShortType, weakTypeTag[types.Integral]), (ByteType, weakTypeTag[types.Integral]),
    (DateType, weakTypeTag[types.Date]), (TimestampType, weakTypeTag[types.DateTime]),
    (StringType, weakTypeTag[types.Text]), (BooleanType, weakTypeTag[types.Binary]),
    (VectorType, weakTypeTag[types.OPVector])
  )

  val sparkCollectionTypeToTypeTagMappings = Seq(
    (ArrayType(LongType, containsNull = true), weakTypeTag[types.DateList]),
    (ArrayType(DoubleType, containsNull = true), weakTypeTag[types.Geolocation]),
    (MapType(StringType, StringType, valueContainsNull = true), weakTypeTag[types.TextMap]),
    (MapType(StringType, DoubleType, valueContainsNull = true), weakTypeTag[types.RealMap]),
    (MapType(StringType, LongType, valueContainsNull = true), weakTypeTag[types.IntegralMap]),
    (MapType(StringType, BooleanType, valueContainsNull = true), weakTypeTag[types.BinaryMap]),
    (MapType(StringType, ArrayType(StringType, containsNull = true), valueContainsNull = true),
      weakTypeTag[types.MultiPickListMap]),
    (MapType(StringType, ArrayType(DoubleType, containsNull = true), valueContainsNull = true),
      weakTypeTag[types.GeolocationMap])
  )

  val sparkNonNullableTypeToTypeTagMappings = Seq(
    (DoubleType, weakTypeTag[types.Real]), (FloatType, weakTypeTag[types.Real])
  )

  Spec(FeatureSparkTypes.getClass) should "assign appropriate feature type tags for valid types" in {
    sparkTypeToTypeTagMappings.foreach(mapping =>
      FeatureSparkTypes.featureTypeTagOf(mapping._1, isNullable = false) shouldBe mapping._2
    )
  }

  it should "assign appropriate feature type tags for valid non-nullable types" in {
    sparkNonNullableTypeToTypeTagMappings.foreach(mapping =>
      FeatureSparkTypes.featureTypeTagOf(mapping._1, isNullable = true) shouldBe mapping._2
    )
  }

  it should "assign appropriate feature type tags for collection types" in {
    sparkCollectionTypeToTypeTagMappings.foreach(mapping =>
      FeatureSparkTypes.featureTypeTagOf(mapping._1, isNullable = true) shouldBe mapping._2
    )
  }

  it should "throw error for unsupported types" in {
    val error = intercept[IllegalArgumentException](FeatureSparkTypes.featureTypeTagOf(BinaryType, isNullable = false))
    error.getMessage shouldBe "Spark BinaryType is currently not supported."
  }

  it should "throw error for unknown types" in {
    val unknownType = NullType
    val error = intercept[IllegalArgumentException](FeatureSparkTypes.featureTypeTagOf(unknownType, isNullable = false))
    error.getMessage shouldBe s"No feature type tag mapping for Spark type $unknownType"
  }

}
