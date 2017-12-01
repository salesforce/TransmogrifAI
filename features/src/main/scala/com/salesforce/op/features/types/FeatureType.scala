/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
  def isFeatureValueType(t: TypeTag[_]): Boolean = FeatureType.featureValueTypeTags.contains(t.tpe.dealias.toString)

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
    classOf[GeolocationMap] -> typeTag[GeolocationMap],
    // Numerics
    classOf[Binary] -> typeTag[Binary],
    classOf[Currency] -> typeTag[Currency],
    classOf[Date] -> typeTag[Date],
    classOf[DateTime] -> typeTag[DateTime],
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
    typeTags.map(tag => tag.tpe.dealias.toString -> tag).toMap
  }

}
