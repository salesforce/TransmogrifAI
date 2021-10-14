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

package com.salesforce.op.features.types

import com.salesforce.op.utils.reflection.ReflectionUtils
import org.apache.spark.ml.linalg.Vector

import scala.collection.TraversableOnce
import scala.reflect.runtime.universe._
import scala.util.Try


/**
 * A representation of Feature Value Type container
 */
trait FeatureType extends Serializable {
  /**
   * Feature value type
   */
  type Value

  /**
   * Returns value
   */
  def value: Value

  /**
   * Returns true is the value is empty, false otherwise
   */
  def isEmpty: Boolean

  /**
   * Returns true is the value is nullable, false otherwise
   */
  def isNullable: Boolean = true

  /**
   * Returns value shortcut
   */
  final def v: Value = value

  /**
   * Returns true is the value is non empty, false otherwise
   */
  final def nonEmpty: Boolean = !isEmpty

  /**
   * Returns true if this feature value is non empty and the predicate
   * $p returns true when applied to this feature type value.
   * Otherwise, returns false.
   *
   * @param p the predicate to test
   */
  final def exists(p: Value => Boolean): Boolean = nonEmpty && p(value)

  /**
   * Returns true if this feature value is non empty and contains the specified value $v.
   * Otherwise, returns false.
   *
   * @param v value to test
   */
  final def contains(v: Value): Boolean = exists(_ == v)

  /**
   * Indicates whether some other object is "equal to" this one
   */
  final override def equals(in: Any): Boolean = in match {
    case f: FeatureType => value.equals(f.value)
    case _ => false
  }

  /**
   * Returns a hash code value for the value
   */
  final override def hashCode: Int = value.hashCode

  /**
   * Returns a string representation of this value
   */
  override def toString: String = {
    val s = value match {
      case _ if isEmpty => ""
      case t: TraversableOnce[_] => t.mkString(", ")
      case x => x.toString
    }
    s"${getClass.getSimpleName}($s)"
  }
}


/**
 * Non Nullable mixin
 */
trait NonNullable extends FeatureType {
  final abstract override def isNullable: Boolean = false
}

/**
 * This exception is thrown when an empty or null value is passed into [[NonNullable]] feature type
 *
 * @param c   feature type class
 * @param msg optional message
 */
class NonNullableEmptyException(c: Class[_ <: NonNullable], msg: Option[String] = None)
  extends IllegalArgumentException(
    s"${c.getSimpleName} cannot be empty${msg.map(m => s": $m").getOrElse("")}"
  )

/**
 * Location mixin
 */
trait Location extends FeatureType

/**
 * Represents a feature type that can take on ''at most one'' of some discrete, finite number of values.
 */
trait SingleResponse extends FeatureType with Categorical

/**
 * Represents a feature type that can take on ''zero or more'' of some discrete, finite number of values.
 */
trait MultiResponse extends FeatureType with Categorical

/**
 * Represents a feature type that can only take on values from some discrete, finite number of values.
 */
trait Categorical extends FeatureType


/**
 * Extractor for non empty feature type value
 */
object SomeValue {

  /**
   * Extractor for non empty feature type value
   *
   * @tparam T feature type
   * @param v feature type
   * @return feature type value
   */
  def unapply[T <: FeatureType](v: T): Option[T#Value] = if (v.isEmpty) None else Some(v.value)
}

/**
 * Feature value type related functions
 */
object FeatureType {

  /**
   * Returns feature type name
   *
   * @tparam O feature type
   * @return feature type name
   */
  def typeName[O <: FeatureType : WeakTypeTag]: String = weakTypeTag[O].tpe.typeSymbol.fullName

  /**
   * Returns short feature type name
   *
   * @tparam O feature type
   * @return short feature type name
   */
  def shortTypeName[O <: FeatureType : WeakTypeTag]: String = {
    val tn = FeatureType.typeName[O]
    tn.split("\\.").lastOption.getOrElse(
      throw new IllegalArgumentException(s"Failed to get feature type name from $tn")
    )
  }

  /**
   * Check whether feature's type [[A]] is a subtype of the feature type [[B]]
   *
   * @tparam A feature type
   * @tparam B feature type
   * @return true if [[A]] conforms to [[B]], false otherwise
   */
  def isSubtype[A <: FeatureType : WeakTypeTag, B <: FeatureType : WeakTypeTag]: Boolean =
    implicitly[WeakTypeTag[A]].tpe <:< implicitly[WeakTypeTag[B]].tpe

  /**
   * If this type tag corresponds to one of the feature types (Real, Text, etc)
   *
   * @param t type tag
   * @return true if this type tag corresponds to one of the feature types, false otherwise
   */
  def isFeatureType(t: TypeTag[_]): Boolean = {
    val runtimeClass = Try(t.mirror.runtimeClass(t.tpe.typeSymbol.asClass).asInstanceOf[Class[_ <: FeatureType]])
    runtimeClass.map(FeatureType.featureTypeTags.contains).getOrElse(false)
  }

  /**
   * If this type tag corresponds to one of the feature value types (Real#Value, Text#Value etc)
   *
   * @param t type tag
   * @return true if this type tag corresponds to one of the feature value types, false otherwise
   */
  def isFeatureValueType(t: TypeTag[_]): Boolean =
    FeatureType.featureValueTypeTags.contains(ReflectionUtils.dealisedTypeName(t.tpe))

  /**
   * Feature type tag
   *
   * @param featureClass feature type class
   * @return feature type tag
   * @throws IllegalArgumentException if type tag is not available
   */
  def featureTypeTag(featureClass: Class[_ <: FeatureType]): TypeTag[_ <: FeatureType] =
    featureTypeTags.getOrElse(featureClass,
      throw new IllegalArgumentException(s"Unknown feature type '$featureClass'"))

  /**
   * Feature type tag
   *
   * @param featureClassName feature type class name
   * @return feature type tag
   * @throws IllegalArgumentException if type tag not available
   * @throws ClassNotFoundException if class not found with given name
   */
  def featureTypeTag(featureClassName: String): TypeTag[_ <: FeatureType] =
    featureTypeTag(ReflectionUtils.classForName(featureClassName).asInstanceOf[Class[_ <: FeatureType]])

  /**
   * Feature value type tag
   *
   * @param featureValueTypeName feature value type name
   * @return feature value type tag
   * @throws IllegalArgumentException if type tag not available
   */
  def featureValueTypeTag(featureValueTypeName: String): TypeTag[_] =
    featureValueTypeTags.getOrElse(featureValueTypeName,
      throw new IllegalArgumentException(s"Unknown feature value type '$featureValueTypeName'"))

  /**
   * A map from feature type class to type tag
   */
  private[types] val featureTypeTags: Map[Class[_ <: FeatureType], TypeTag[_ <: FeatureType]] = Map(
    // Vector
    classOf[OPVector] -> typeTag[OPVector],
    // Lists
    classOf[TextList] -> typeTag[TextList],
    classOf[DateList] -> typeTag[DateList],
    classOf[DateTimeList] -> typeTag[DateTimeList],
    classOf[Geolocation] -> typeTag[Geolocation],
    // Maps
    classOf[Base64Map] -> typeTag[Base64Map],
    classOf[BinaryMap] -> typeTag[BinaryMap],
    classOf[ComboBoxMap] -> typeTag[ComboBoxMap],
    classOf[CurrencyMap] -> typeTag[CurrencyMap],
    classOf[DateMap] -> typeTag[DateMap],
    classOf[DateTimeMap] -> typeTag[DateTimeMap],
    classOf[EmailMap] -> typeTag[EmailMap],
    classOf[IDMap] -> typeTag[IDMap],
    classOf[IntegerMap] -> typeTag[IntegerMap],
    classOf[IntegralMap] -> typeTag[IntegralMap],
    classOf[MultiPickListMap] -> typeTag[MultiPickListMap],
    classOf[PercentMap] -> typeTag[PercentMap],
    classOf[PhoneMap] -> typeTag[PhoneMap],
    classOf[PickListMap] -> typeTag[PickListMap],
    classOf[RealMap] -> typeTag[RealMap],
    classOf[TextAreaMap] -> typeTag[TextAreaMap],
    classOf[TextMap] -> typeTag[TextMap],
    classOf[URLMap] -> typeTag[URLMap],
    classOf[CountryMap] -> typeTag[CountryMap],
    classOf[StateMap] -> typeTag[StateMap],
    classOf[CityMap] -> typeTag[CityMap],
    classOf[PostalCodeMap] -> typeTag[PostalCodeMap],
    classOf[StreetMap] -> typeTag[StreetMap],
    classOf[NameStats] -> typeTag[NameStats],
    classOf[GeolocationMap] -> typeTag[GeolocationMap],
    classOf[Prediction] -> typeTag[Prediction],
    // Numerics
    classOf[Binary] -> typeTag[Binary],
    classOf[Currency] -> typeTag[Currency],
    classOf[Date] -> typeTag[Date],
    classOf[DateTime] -> typeTag[DateTime],
    classOf[Integer] -> typeTag[Integer],
    classOf[Integral] -> typeTag[Integral],
    classOf[Percent] -> typeTag[Percent],
    classOf[Real] -> typeTag[Real],
    classOf[RealNN] -> typeTag[RealNN],
    // Sets
    classOf[MultiPickList] -> typeTag[MultiPickList],
    // Text
    classOf[Base64] -> typeTag[Base64],
    classOf[ComboBox] -> typeTag[ComboBox],
    classOf[Email] -> typeTag[Email],
    classOf[ID] -> typeTag[ID],
    classOf[Phone] -> typeTag[Phone],
    classOf[PickList] -> typeTag[PickList],
    classOf[Text] -> typeTag[Text],
    classOf[TextArea] -> typeTag[TextArea],
    classOf[URL] -> typeTag[URL],
    classOf[Country] -> typeTag[Country],
    classOf[State] -> typeTag[State],
    classOf[City] -> typeTag[City],
    classOf[PostalCode] -> typeTag[PostalCode],
    classOf[Street] -> typeTag[Street]
  )

  /**
   * A map from feature values type to type tag
   */
  private[types] val featureValueTypeTags: Map[String, TypeTag[_]] = {
    val typeTags = Seq(
      // Vector
      typeTag[Vector],
      // Lists
      typeTag[Seq[Double]],
      typeTag[Seq[Long]],
      typeTag[Seq[String]],
      // Maps
      typeTag[Map[String, Boolean]],
      typeTag[Map[String, Double]],
      typeTag[Map[String, Long]],
      typeTag[Map[String, String]],
      typeTag[Map[String, Seq[Double]]],
      typeTag[Map[String, Set[String]]],
      // Numerics
      typeTag[Option[Boolean]],
      typeTag[Option[Double]],
      typeTag[Option[Long]],
      // Sets
      typeTag[Set[String]],
      // Text
      typeTag[Option[String]]
    )
    typeTags.map(tag => ReflectionUtils.dealisedTypeName(tag.tpe) -> tag).toMap
  }

}
