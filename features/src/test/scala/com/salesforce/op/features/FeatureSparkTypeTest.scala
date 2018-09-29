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

import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.junit.runner.RunWith
import org.scalatest.{Assertion, FlatSpec}
import org.scalatest.junit.JUnitRunner

import scala.reflect.runtime.universe._

@RunWith(classOf[JUnitRunner])
class FeatureSparkTypeTest extends FlatSpec with TestSparkContext {
  val primitiveTypes = Seq(
    (DoubleType, weakTypeTag[types.Real], DoubleType),
    (FloatType, weakTypeTag[types.Real], DoubleType),
    (LongType, weakTypeTag[types.Integral], LongType),
    (IntegerType, weakTypeTag[types.Integral], LongType),
    (ShortType, weakTypeTag[types.Integral], LongType),
    (ByteType, weakTypeTag[types.Integral], LongType),
    (DateType, weakTypeTag[types.Date], LongType),
    (TimestampType, weakTypeTag[types.DateTime], LongType),
    (StringType, weakTypeTag[types.Text], StringType),
    (BooleanType, weakTypeTag[types.Binary], BooleanType),
    (VectorType, weakTypeTag[types.OPVector], VectorType)
  )

  val nonNullable = Seq(
    (DoubleType, weakTypeTag[types.RealNN], DoubleType),
    (FloatType, weakTypeTag[types.RealNN], DoubleType)
  )

  private def mapType(v: DataType) = MapType(StringType, v, valueContainsNull = true)
  private def arrType(v: DataType) = ArrayType(v, containsNull = true)

  val collectionTypes = Seq(
    (arrType(LongType), weakTypeTag[types.DateList], arrType(LongType)),
    (arrType(DoubleType), weakTypeTag[types.Geolocation], arrType(DoubleType)),
    (arrType(StringType), weakTypeTag[types.TextList], arrType(StringType)),
    (mapType(StringType), weakTypeTag[types.TextMap], mapType(StringType)),
    (mapType(DoubleType), weakTypeTag[types.RealMap], mapType(DoubleType)),
    (mapType(LongType), weakTypeTag[types.IntegralMap], mapType(LongType)),
    (mapType(BooleanType), weakTypeTag[types.BinaryMap], mapType(BooleanType)),
    (mapType(arrType(StringType)), weakTypeTag[types.MultiPickListMap], mapType(arrType(StringType))),
    (mapType(arrType(DoubleType)), weakTypeTag[types.GeolocationMap], mapType(arrType(DoubleType)))
  )

  Spec(FeatureSparkTypes.getClass) should "assign appropriate feature type tags for valid types and versa" in {
    primitiveTypes.map(scala.Function.tupled(assertTypes()))
  }

  it should "assign appropriate feature type tags for valid non-nullable types and versa" in {
    nonNullable.map(scala.Function.tupled(assertTypes(isNullable = false)))
  }

  it should "assign appropriate feature type tags for collection types and versa" in {
    collectionTypes.map(scala.Function.tupled(assertTypes()))
  }

  it should "error for unsupported types" in {
    val error = intercept[IllegalArgumentException](FeatureSparkTypes.featureTypeTagOf(BinaryType, isNullable = false))
    error.getMessage shouldBe "Spark BinaryType is currently not supported."
  }

  it should "error for unknown types" in {
    val unknownType = NullType
    val error = intercept[IllegalArgumentException](FeatureSparkTypes.featureTypeTagOf(unknownType, isNullable = false))
    error.getMessage shouldBe s"No feature type tag mapping for Spark type $unknownType"
  }

  def assertTypes(
    isNullable: Boolean = true
  )(
    sparkType: DataType,
    featureType: WeakTypeTag[_ <: FeatureType],
    expectedSparkType: DataType
  ): Assertion = {
    FeatureSparkTypes.featureTypeTagOf(sparkType, isNullable) shouldBe featureType
    FeatureSparkTypes.sparkTypeOf(featureType) shouldBe expectedSparkType
  }

}
