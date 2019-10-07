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

import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.features.{types => t}
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichDataType._
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.column
import org.apache.spark.sql.types.{StructType, _}
import org.apache.spark.sql.{Column, Encoder, Row, TypedColumn}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._

/**
 * Feature Spark types mappings
 */
case object FeatureSparkTypes {

  // Numerics
  val Binary = BooleanType
  val Integral = LongType
  val Real = DoubleType
  val RealNN = Real
  val Date = Integral
  val DateTime = Integral
  val Currency = Real
  val Percent = Real

  // Text
  val Text = StringType
  val Base64 = Text
  val ComboBox = Text
  val Email = Text
  val ID = Text
  val Phone = Text
  val PickList = Text
  val TextArea = Text
  val URL = Text
  val Country = Text
  val State = Text
  val City = Text
  val PostalCode = Text
  val Street = Text
  val Name = Text

  // Vector
  val OPVector = VectorType

  // Lists
  val TextList = ArrayType(Text, containsNull = true)
  val NameList = ArrayType(Name, containsNull = true)
  val DateList = ArrayType(Date, containsNull = true)
  val DateTimeList = DateList
  val Geolocation = ArrayType(Real, containsNull = true)

  // Sets
  val MultiPickList = ArrayType(Text, containsNull = true)

  // Maps
  private def mapType(v: DataType) = MapType(StringType, v, valueContainsNull = true)

  val Base64Map = mapType(Base64)
  val BinaryMap = mapType(Binary)
  val ComboBoxMap = mapType(ComboBox)
  val CurrencyMap = mapType(Currency)
  val DateMap = mapType(Date)
  val DateTimeMap = mapType(DateTime)
  val EmailMap = mapType(Email)
  val IDMap = mapType(ID)
  val IntegralMap = mapType(Integral)
  val MultiPickListMap = mapType(MultiPickList)
  val PercentMap = mapType(Percent)
  val PhoneMap = mapType(Phone)
  val PickListMap = mapType(PickList)
  val RealMap = mapType(Real)
  val TextAreaMap = mapType(TextArea)
  val TextMap = mapType(Text)
  val URLMap = mapType(URL)
  val CountryMap = mapType(Country)
  val StateMap = mapType(State)
  val CityMap = mapType(City)
  val PostalCodeMap = mapType(PostalCode)
  val StreetMap = mapType(Street)
  val GeolocationMap = mapType(Geolocation)
  val Prediction = mapType(Real)

  /**
   * Spark type of a feature type
   *
   * @tparam O feature type
   * @return spark type
   */
  def sparkTypeOf[O <: FeatureType : WeakTypeTag]: DataType = weakTypeOf[O] match {
    // Vector
    case wt if wt =:= weakTypeOf[t.OPVector] => OPVector

    // Lists
    case wt if wt =:= weakTypeOf[t.TextList] => TextList
      case wt if wt =:= weakTypeOf[t.NameList] => NameList
    case wt if wt =:= weakTypeOf[t.DateList] => DateList
    case wt if wt =:= weakTypeOf[t.DateTimeList] => DateTimeList
    case wt if wt =:= weakTypeOf[t.Geolocation] => Geolocation

    // Maps
    case wt if wt =:= weakTypeOf[t.Base64Map] => Base64Map
    case wt if wt =:= weakTypeOf[t.BinaryMap] => BinaryMap
    case wt if wt =:= weakTypeOf[t.ComboBoxMap] => ComboBoxMap
    case wt if wt =:= weakTypeOf[t.CurrencyMap] => CurrencyMap
    case wt if wt =:= weakTypeOf[t.DateMap] => DateMap
    case wt if wt =:= weakTypeOf[t.DateTimeMap] => DateTimeMap
    case wt if wt =:= weakTypeOf[t.EmailMap] => EmailMap
    case wt if wt =:= weakTypeOf[t.IDMap] => IDMap
    case wt if wt =:= weakTypeOf[t.IntegralMap] => IntegralMap
    case wt if wt =:= weakTypeOf[t.MultiPickListMap] => MultiPickListMap
    case wt if wt =:= weakTypeOf[t.PercentMap] => PercentMap
    case wt if wt =:= weakTypeOf[t.PhoneMap] => PhoneMap
    case wt if wt =:= weakTypeOf[t.PickListMap] => PickListMap
    case wt if wt =:= weakTypeOf[t.RealMap] => RealMap
    case wt if wt =:= weakTypeOf[t.TextAreaMap] => TextAreaMap
    case wt if wt =:= weakTypeOf[t.TextMap] => TextMap
    case wt if wt =:= weakTypeOf[t.URLMap] => URLMap
    case wt if wt =:= weakTypeOf[t.CountryMap] => CountryMap
    case wt if wt =:= weakTypeOf[t.StateMap] => StateMap
    case wt if wt =:= weakTypeOf[t.CityMap] => CityMap
    case wt if wt =:= weakTypeOf[t.PostalCodeMap] => PostalCodeMap
    case wt if wt =:= weakTypeOf[t.StreetMap] => StreetMap
    case wt if wt =:= weakTypeOf[t.GeolocationMap] => GeolocationMap
    case wt if wt =:= weakTypeOf[t.Prediction] => Prediction

    // Numerics
    case wt if wt =:= weakTypeOf[t.Binary] => Binary
    case wt if wt =:= weakTypeOf[t.Currency] => Currency
    case wt if wt =:= weakTypeOf[t.Date] => Date
    case wt if wt =:= weakTypeOf[t.DateTime] => DateTime
    case wt if wt =:= weakTypeOf[t.Integral] => Integral
    case wt if wt =:= weakTypeOf[t.Percent] => Percent
    case wt if wt =:= weakTypeOf[t.Real] => Real
    case wt if wt =:= weakTypeOf[t.RealNN] => RealNN

    // Sets
    case wt if wt =:= weakTypeOf[t.MultiPickList] => MultiPickList

    // Text
    case wt if wt =:= weakTypeOf[t.Base64] => Base64
    case wt if wt =:= weakTypeOf[t.ComboBox] => ComboBox
    case wt if wt =:= weakTypeOf[t.Email] => Email
    case wt if wt =:= weakTypeOf[t.ID] => ID
    case wt if wt =:= weakTypeOf[t.Phone] => Phone
    case wt if wt =:= weakTypeOf[t.PickList] => PickList
    case wt if wt =:= weakTypeOf[t.Text] => Text
    case wt if wt =:= weakTypeOf[t.TextArea] => TextArea
    case wt if wt =:= weakTypeOf[t.URL] => URL
    case wt if wt =:= weakTypeOf[t.Country] => Country
    case wt if wt =:= weakTypeOf[t.State] => State
    case wt if wt =:= weakTypeOf[t.PostalCode] => PostalCode
    case wt if wt =:= weakTypeOf[t.City] => City
    case wt if wt =:= weakTypeOf[t.Street] => Street

    // Unknown
    case wt => throw new IllegalArgumentException(s"No Spark type mapping for feature type $wt")
  }

  /**
   * Feature type TypeTag of a Spark type
   *
   * @param sparkType  spark type
   * @param isNullable is nullable
   * @return feature type TypeTag
   */
  def featureTypeTagOf(sparkType: DataType, isNullable: Boolean): WeakTypeTag[_ <: FeatureType] = sparkType match {
    case DoubleType if !isNullable => weakTypeTag[types.RealNN]
    case DoubleType => weakTypeTag[types.Real]
    case FloatType if !isNullable => weakTypeTag[types.RealNN]
    case FloatType => weakTypeTag[types.Real]
    case ByteType => weakTypeTag[types.Integral]
    case ShortType => weakTypeTag[types.Integral]
    case IntegerType => weakTypeTag[types.Integral]
    case LongType => weakTypeTag[types.Integral]
    case DateType => weakTypeTag[types.Date]
    case TimestampType => weakTypeTag[types.DateTime]
    case ArrayType(StringType, _) => weakTypeTag[types.TextList]
    case StringType => weakTypeTag[types.Text]
    case BooleanType => weakTypeTag[types.Binary]
    case ArrayType(LongType, _) => weakTypeTag[types.DateList]
    case ArrayType(DoubleType, _) => weakTypeTag[types.Geolocation]
    case MapType(StringType, StringType, _) => weakTypeTag[types.TextMap]
    case MapType(StringType, DoubleType, _) => weakTypeTag[types.RealMap]
    case MapType(StringType, LongType, _) => weakTypeTag[types.IntegralMap]
    case MapType(StringType, BooleanType, _) => weakTypeTag[types.BinaryMap]
    case MapType(StringType, ArrayType(StringType, _), _) => weakTypeTag[types.MultiPickListMap]
    case MapType(StringType, ArrayType(DoubleType, _), _) => weakTypeTag[types.GeolocationMap]
    case VectorType => weakTypeTag[types.OPVector]
    case BinaryType => throw new IllegalArgumentException("Spark BinaryType is currently not supported")
    case _ => throw new IllegalArgumentException(s"No feature type tag mapping for Spark type $sparkType")
  }

  private val MapArrStrEncoder = ExpressionEncoder()(typeTag[Map[String, Array[String]]])
  private val MapArrDoubleEncoder = ExpressionEncoder()(typeTag[Map[String, Array[Double]]])
  private val ArrStrEncoder = ExpressionEncoder()(typeTag[Array[String]])

  /**
   * Returns an encoder for a feature type value
   *
   * @param ttv feature type value TypeTag
   * @tparam O feature type
   * @return an encoder for a feature type value
   */
  def featureTypeEncoder[O <: FeatureType : WeakTypeTag](implicit ttv: TypeTag[O#Value]): Encoder[O#Value] = {
    val encoder = weakTypeOf[O] match {
      // Maps
      case wt if wt <:< weakTypeOf[t.MultiPickListMap] => MapArrStrEncoder
      case wt if wt <:< weakTypeOf[t.GeolocationMap] => MapArrDoubleEncoder

      // Sets
      case wt if wt <:< weakTypeOf[t.MultiPickList] => ArrStrEncoder

      // Everything else
      case _ => ExpressionEncoder()(ReflectionUtils.dealiasedTypeTagForType[O#Value]()(ttv))
    }
    encoder.asInstanceOf[Encoder[O#Value]]
  }

  /**
   * Creates a Spark UDF with given function I => O
   *
   * @param f function I => O
   * @tparam I input type
   * @tparam O output type
   * @return a Spark UDF
   */
  def udf1[I <: FeatureType : TypeTag, O <: FeatureType : TypeTag](
    f: I => O
  ): UserDefinedFunction = {
    val inputTypes = Some(FeatureSparkTypes.sparkTypeOf[I] :: Nil)
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    val func = transform1[I, O](f)
    UserDefinedFunction(func, outputType, inputTypes)
  }

  /**
   * Creates a transform function suitable for Spark types with given function I => O
   *
   * @param f function I => O
   * @tparam I input type
   * @tparam O output type
   * @return transform function
   */
  def transform1[I <: FeatureType : TypeTag, O <: FeatureType](
    f: I => O
  ): Any => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convert = FeatureTypeSparkConverter[I]()
    (in1: Any) => {
      val i1: I = convert.fromSpark(in1)
      FeatureTypeSparkConverter.toSpark(f(i1))
    }
  }

  /**
   * Creates a Spark UDF with given function (I1, I2) => O
   *
   * @param f function (I1, I2) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam O  output type
   * @return a Spark UDF
   */
  def udf2[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, O <: FeatureType : TypeTag](
    f: (I1, I2) => O
  ): UserDefinedFunction = {
    val inputTypes = Some(FeatureSparkTypes.sparkTypeOf[I1] :: FeatureSparkTypes.sparkTypeOf[I2] :: Nil)
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    val func = transform2[I1, I2, O](f)
    UserDefinedFunction(func, outputType, inputTypes)
  }

  /**
   * Creates a transform function suitable for Spark types with given function (I1, I2) => O
   *
   * @param f function (I1, I2) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam O  output type
   * @return transform function
   */
  def transform2[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, O <: FeatureType](
    f: (I1, I2) => O
  ): (Any, Any) => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convertI1 = FeatureTypeSparkConverter[I1]()
    val convertI2 = FeatureTypeSparkConverter[I2]()
    (in1: Any, in2: Any) => {
      val i1: I1 = convertI1.fromSpark(in1)
      val i2: I2 = convertI2.fromSpark(in2)
      FeatureTypeSparkConverter.toSpark(f(i1, i2))
    }
  }

  /**
   * Creates a Spark UDF with given function (I1, I2, I3) => O
   *
   * @param f function (I1, I2, I3) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam I3 3rd input type
   * @tparam O  output type
   * @return a Spark UDF
   */
  def udf3[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, I3 <: FeatureType : TypeTag,
  O <: FeatureType : TypeTag](
    f: (I1, I2, I3) => O
  ): UserDefinedFunction = {
    val inputTypes = Some(
      FeatureSparkTypes.sparkTypeOf[I1] :: FeatureSparkTypes.sparkTypeOf[I2] ::
        FeatureSparkTypes.sparkTypeOf[I3] :: Nil
    )
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    val func = transform3[I1, I2, I3, O](f)
    UserDefinedFunction(func, outputType, inputTypes)
  }

  /**
   * Creates a transform function suitable for Spark types with given function (I1, I2, I3) => O
   *
   * @param f function (I1, I2, I3) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam I3 3rd input type
   * @tparam O  output type
   * @return transform function
   */
  def transform3[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, I3 <: FeatureType : TypeTag,
  O <: FeatureType](
    f: (I1, I2, I3) => O
  ): (Any, Any, Any) => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convertI1 = FeatureTypeSparkConverter[I1]()
    val convertI2 = FeatureTypeSparkConverter[I2]()
    val convertI3 = FeatureTypeSparkConverter[I3]()
    (in1: Any, in2: Any, in3: Any) => {
      val i1: I1 = convertI1.fromSpark(in1)
      val i2: I2 = convertI2.fromSpark(in2)
      val i3: I3 = convertI3.fromSpark(in3)
      FeatureTypeSparkConverter.toSpark(f(i1, i2, i3))
    }
  }

  /**
   * Creates a Spark UDF with given function (I1, I2, I3, I4) => O
   *
   * @param f function (I1, I2, I3, I4) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam I3 3rd input type
   * @tparam I4 4th input type
   * @tparam O  output type
   * @return a Spark UDF
   */
  def udf4[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, I3 <: FeatureType : TypeTag,
  I4 <: FeatureType : TypeTag, O <: FeatureType : TypeTag](
    f: (I1, I2, I3, I4) => O
  ): UserDefinedFunction = {
    val inputTypes = Some(
      FeatureSparkTypes.sparkTypeOf[I1] :: FeatureSparkTypes.sparkTypeOf[I2] ::
        FeatureSparkTypes.sparkTypeOf[I3] :: FeatureSparkTypes.sparkTypeOf[I4] :: Nil
    )
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    val func = transform4[I1, I2, I3, I4, O](f)
    UserDefinedFunction(func, outputType, inputTypes)
  }

  /**
   * Creates a transform function suitable for Spark types with given function (I1, I2, I3, I4) => O
   *
   * @param f function (I1, I2, I3, I4) => O
   * @tparam I1 1st input type
   * @tparam I2 2nd input type
   * @tparam I3 3rd input type
   * @tparam I4 4th input type
   * @tparam O  output type
   * @return transform function
   */
  def transform4[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, I3 <: FeatureType : TypeTag,
  I4 <: FeatureType : TypeTag, O <: FeatureType](
    f: (I1, I2, I3, I4) => O
  ): (Any, Any, Any, Any) => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convertI1 = FeatureTypeSparkConverter[I1]()
    val convertI2 = FeatureTypeSparkConverter[I2]()
    val convertI3 = FeatureTypeSparkConverter[I3]()
    val convertI4 = FeatureTypeSparkConverter[I4]()
    (in1: Any, in2: Any, in3: Any, in4: Any) => {
      val i1: I1 = convertI1.fromSpark(in1)
      val i2: I2 = convertI2.fromSpark(in2)
      val i3: I3 = convertI3.fromSpark(in3)
      val i4: I4 = convertI4.fromSpark(in4)
      FeatureTypeSparkConverter.toSpark(f(i1, i2, i3, i4))
    }
  }

  /**
   * Creates a Spark UDF with given function Seq[I] => O
   *
   * @param f function Seq[I] => O
   * @tparam I input type
   * @tparam O output type
   * @return a Spark UDF
   */
  def udfN[I <: FeatureType : TypeTag, O <: FeatureType : TypeTag](
    f: Seq[I] => O
  ): UserDefinedFunction = {
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convert = FeatureTypeSparkConverter[I]()
    val func = (r: Row) => {
      val arr = new ArrayBuffer[I](r.length)
      var i = 0
      while (i < r.length) {
        arr += convert.fromSpark(r.get(i))
        i += 1
      }
      FeatureTypeSparkConverter.toSpark(f(arr))
    }
    UserDefinedFunction(func, outputType, inputTypes = None)
  }

  /**
   * Creates a transform function suitable for Spark types with given function Seq[I] => O
   *
   * @param f function Seq[I] => O
   * @tparam I input type
   * @tparam O output type
   * @return transform function
   */
  def transformN[I <: FeatureType : TypeTag, O <: FeatureType](
    f: Seq[I] => O
  ): Array[Any] => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convert = FeatureTypeSparkConverter[I]()
    (r: Array[Any]) => {
      val arr = new ArrayBuffer[I](r.length)
      var i = 0
      while (i < r.length) {
        arr += convert.fromSpark(r(i))
        i += 1
      }
      FeatureTypeSparkConverter.toSpark(f(arr))
    }
  }

  /**
   * Creates a Spark UDF with given function (I1, Seq[I2]) => O
   *
   * @param f function (I1, Seq[I2]) => O
   * @tparam I1 input of singular type
   * @tparam I2 input of sequence type
   * @tparam O output type
   * @return a Spark UDF
   */
  def udf2N[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, O <: FeatureType : TypeTag]
  (
    f: (I1, Seq[I2]) => O
  ): UserDefinedFunction = {
    val outputType = FeatureSparkTypes.sparkTypeOf[O]
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convertI1 = FeatureTypeSparkConverter[I1]()
    val convertI2 = FeatureTypeSparkConverter[I2]()
    val func = (r: Row) => {
      val arr = new ArrayBuffer[I2](r.length - 1)
      val i1: I1 = convertI1.fromSpark(r.get(0))
      var i = 1
      while (i < r.length) {
        arr += convertI2.fromSpark(r.get(i))
        i += 1
      }
      FeatureTypeSparkConverter.toSpark(f(i1, arr))
    }
    UserDefinedFunction(func, outputType, inputTypes = None)
  }

  /**
   * Creates a transform function suitable for Spark types with given function (I1, Seq[I2]) => O
   *
   * @param f function (I1, Seq[I2]) => O
   * @tparam I1 input of singular type
   * @tparam I2 input of sequence type
   * @tparam O output type
   * @return transform function
   */
  def transform2N[I1 <: FeatureType : TypeTag, I2 <: FeatureType : TypeTag, O <: FeatureType: TypeTag]
  (
    f: (I1, Seq[I2]) => O
  ): (Any, Array[Any]) => Any = {
    // Converters MUST be defined outside the result function since they involve reflection calls
    val convertI1 = FeatureTypeSparkConverter[I1]()
    val convertI2 = FeatureTypeSparkConverter[I2]()
    (in1: Any, r: Array[Any]) => {
      val i1: I1 = convertI1.fromSpark(in1)
      val arr = new ArrayBuffer[I2](r.length)
      var i = 0
      while (i < r.length) {
        arr += convertI2.fromSpark(r(i))
        i += 1
      }
      FeatureTypeSparkConverter.toSpark(f(i1, arr))
    }
  }

  /**
   * Create a sql [[Column]] instance of a feature
   */
  def toColumn(f: OPFeature): Column = column(f.name)

  /**
   * Create a sql [[TypedColumn]] instance of a feature
   */
  def toTypedColumn[T <: FeatureType](f: FeatureLike[T])(implicit e: Encoder[T#Value]): TypedColumn[Any, T#Value] = {
    toColumn(f).as[T#Value](e)
  }

  /**
   * Create a sql schema [[StructType]] from a list of features
   *
   * @param features features
   * @return struct type
   */
  def toStructType(features: OPFeature*): StructType = StructType(features.map(toStructField(_)))

  /**
   * Create a sql column type [[StructField]] for a given feature
   *
   * @param f        feature
   * @param metadata feature metadata
   * @return struct field
   */
  def toStructField(f: OPFeature, metadata: Metadata = Metadata.empty): StructField = {
    StructField(name = f.name, dataType = sparkTypeOf(f.wtt), nullable = true, metadata = metadata)
  }

  /**
   * Validate sql schema against the specified features
   *
   * @param schema   sql schema [[StructType]]
   * @param features list of features
   */
  def validateSchema(schema: StructType, features: Seq[OPFeature]): Seq[String] = {
    val fieldsMap = schema.fields.map(f => f.name -> f).toMap

    val validationResults =
      for {
        feature <- features
        featureSchema = toStructField(feature)
      } yield fieldsMap.get(featureSchema.name) match {
        case None => Some(s"No column named '${featureSchema.name}'")
        case Some(field) if field.name != featureSchema.name ||
          !field.dataType.equalsIgnoreNullability(featureSchema.dataType) =>
          // nullable is not check in order to avoid confusion between some
          // usual schemas and UDF schema
          Some(s"Column and feature types don't match: $field != $featureSchema")
        case _ => None
      }

    validationResults.flatten
  }

}
