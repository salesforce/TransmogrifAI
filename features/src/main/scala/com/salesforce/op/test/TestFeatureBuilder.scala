/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.test

import com.salesforce.op.UID
import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter, Real}
import com.salesforce.op.features.{Feature, FeatureBuilder, FeatureSparkTypes}
import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.StructType

import scala.reflect.runtime.universe._

/**
 * Test Feature Builder is a factory for creating datasets and features for tests
 */
case object TestFeatureBuilder {

  case object DefaultFeatureNames {
    val (f1, f2, f3, f4, f5) = ("f1", "f2", "f3", "f4", "f5")
  }

  private val dummyFeature = feature[Real]("dummy")
  private lazy val DefaultFeatureArgs = ReflectionUtils.bestCtorWithArgs(dummyFeature)._2.toMap

  /**
   * Build features from a given dataset
   *
   * @param ds          dataset
   * @param nonNullable non nullable fields
   * @return array of features
   */
  def apply(ds: DataFrame, nonNullable: Set[String]): Array[Feature[_ <: FeatureType]] = apply(ds.schema, nonNullable)

  /**
   * Build features from a given schema
   *
   * @param schema      schema
   * @param nonNullable non nullable fields
   * @return array of features
   */
  def apply(schema: StructType, nonNullable: Set[String]): Array[Feature[_ <: FeatureType]] = {
    val featureClass = classOf[Feature[_ <: FeatureType]]

    schema.fields.map(field => {
      val isNullable = !nonNullable.contains(field.name)
      def ctorArgs(argName: String, argSymbol: Symbol): scala.util.Try[Any] = scala.util.Try {
        argName match {
          case "name" => field.name
          case "uid" => UID(featureClass)
          case "wtt" => FeatureSparkTypes.featureTypeTagOf(field.dataType, isNullable)
          case n => DefaultFeatureArgs(n)
        }
      }
      ReflectionUtils.newInstance[Feature[_ <: FeatureType]](featureClass, ctorArgs)
    })
  }

  /**
   * Build a dataset with one feature of specified type
   *
   * @param f1name feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 feature type
   * @return dataset with one feature of specified type
   */
  def apply[F1 <: FeatureType : TypeTag](
    f1name: String, data: Seq[F1]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1]) = {
    val f1 = feature[F1](f1name)
    val schema = FeatureSparkTypes.toStructType(f1)
    (dataframe(schema, data.map(Tuple1(_))), f1)
  }

  /**
   * Build a dataset with one feature of specified type
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 feature type
   * @return dataset with one feature of specified type
   */
  def apply[F1 <: FeatureType : TypeTag](data: Seq[F1])(implicit spark: SparkSession): (DataFrame, Feature[F1]) = {
    apply[F1](f1name = DefaultFeatureNames.f1, data)
  }

  /**
   * Build a dataset with two features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @return dataset with two features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag](
    f1name: String, f2name: String, data: Seq[(F1, F2)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2]) = {
    val (f1, f2) = (feature[F1](f1name), feature[F2](f2name))
    val schema = FeatureSparkTypes.toStructType(f1, f2)
    (dataframe(schema, data), f1, f2)
  }

  /**
   * Build a dataset with two features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @return dataset with two features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag](data: Seq[(F1, F2)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2]) = {
    apply[F1, F2](f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2, data)
  }

  /**
   * Build a dataset with three features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @return dataset with three features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag, F3 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, data: Seq[(F1, F2, F3)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3]) = {
    val (f1, f2, f3) = (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3)
    (dataframe(schema, data), f1, f2, f3)
  }

  /**
   * Build a dataset with three features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @return dataset with three features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag, F3 <: FeatureType : TypeTag](
    data: Seq[(F1, F2, F3)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3]) = {
    apply[F1, F2, F3](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, data)
  }

  /**
   * Build a dataset with four features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param f4name 4th feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @return dataset with four features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, f4name: String, data: Seq[(F1, F2, F3, F4)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4]) = {
    val (f1, f2, f3, f4) = (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name), feature[F4](f4name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3, f4)
    (dataframe(schema, data), f1, f2, f3, f4)
  }

  /**
   * Build a dataset with four features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @return dataset with four features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag](data: Seq[(F1, F2, F3, F4)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4]) = {
    apply[F1, F2, F3, F4](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, f4name = DefaultFeatureNames.f4, data)
  }

  /**
   * Build a dataset with five features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param f4name 4th feature name
   * @param f5name 5th feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @tparam F5 5th feature type
   * @return dataset with five features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag,
  F5 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, f4name: String, f5name: String, data: Seq[(F1, F2, F3, F4, F5)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4], Feature[F5]) = {
    val (f1, f2, f3, f4, f5) =
      (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name), feature[F4](f4name), feature[F5](f5name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3, f4, f5)
    (dataframe(schema, data), f1, f2, f3, f4, f5)
  }

  /**
   * Build a dataset with five features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @tparam F5 5th feature type
   * @return dataset with five features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag,
  F5 <: FeatureType : TypeTag](data: Seq[(F1, F2, F3, F4, F5)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4], Feature[F5]) = {
    apply[F1, F2, F3, F4, F5](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, f4name = DefaultFeatureNames.f4,
      f5name = DefaultFeatureNames.f5, data)
  }

  private def dataframe[T <: Product](schema: StructType, data: Seq[T])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    implicit val rowEncoder = RowEncoder(schema)

    data.map(p => Row.fromSeq(
      p.productIterator.toSeq.map { case f: FeatureType => FeatureTypeSparkConverter.toSpark(f) }
    )).toDF()
  }

  private def feature[T <: FeatureType](name: String)(implicit tt: TypeTag[T]) =
    FeatureBuilder.fromRow[T](name)(tt).asPredictor
}
