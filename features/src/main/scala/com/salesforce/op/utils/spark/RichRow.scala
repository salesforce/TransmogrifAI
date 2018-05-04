/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.features.types.{FeatureType, FeatureTypeSparkConverter}
import com.salesforce.op.features.{FeatureLike, TransientFeature}
import org.apache.spark.sql.Row


/**
 * [[org.apache.spark.sql.Row]] enrichment functions
 */
object RichRow {

  implicit class RichRow(val row: Row) extends AnyVal {

    /**
     * Returns the value at position i. If the value is null, null is returned. The following
     * is a mapping between Spark SQL types and return types:
     *
     * {{{
     *   BooleanType -> java.lang.Boolean
     *   ByteType -> java.lang.Byte
     *   ShortType -> java.lang.Short
     *   IntegerType -> java.lang.Integer
     *   FloatType -> java.lang.Float
     *   DoubleType -> java.lang.Double
     *   StringType -> String
     *   DecimalType -> java.math.BigDecimal
     *
     *   DateType -> java.sql.Date
     *   TimestampType -> java.sql.Timestamp
     *
     *   BinaryType -> byte array
     *   ArrayType -> scala.collection.Seq (use getList for java.util.List)
     *   MapType -> scala.collection.Map (use getJavaMap for java.util.Map)
     *   StructType -> org.apache.spark.sql.Row
     * }}}
     */
    def getAny(fieldName: String): Any = row.get(row.fieldIndex(fieldName))

    /**
     * Returns map feature by name
     * @param fieldName name of map feature
     * @return feature value as instance of Map[String, Any]
     */
    def getMapAny(fieldName: String): scala.collection.Map[String, Any] =
      row.getMap[String, Any](row.fieldIndex(fieldName))

    /**
     * Returns the value of field named {fieldName}. If the value is null, None is returned.
     */
    def getOptionAny(fieldName: String): Option[Any] = Option(getAny(fieldName))

    /**
     * Returns the value at position i. If the value is null, None is returned.
     */
    def getOptionAny(i: Integer): Option[Any] = Option(row.get(i))

    /**
     * Returns the value of field named {fieldName}. If the value is null, None is returned.
     */
    def getOption[T](fieldName: String): Option[T] = getOptionAny(fieldName) collect { case t: T @unchecked => t }

    /**
     * Returns the value at position i. If the value is null, None is returned.
     */
    def getOption[T](i: Integer): Option[T] = getOptionAny(i) collect { case t: T @unchecked => t }

    /**
     * Returns the value of a given feature casted into the feature type
     *
     * @throws UnsupportedOperationException when schema is not defined.
     * @throws IllegalArgumentException when fieldName do not exist.
     * @throws ClassCastException when data type does not match.
     */
    def getFeatureType[T <: FeatureType](f: FeatureLike[T])(implicit conv: FeatureTypeSparkConverter[T]): T =
      conv.fromSpark(getAny(f.name))

    /**
     * Returns the value of a given feature casted into the feature type from the transient feature and the
     * weak type tag of features
     *
     * @throws UnsupportedOperationException when schema is not defined.
     * @throws IllegalArgumentException when fieldName do not exist.
     * @throws ClassCastException when data type does not match.
     */
    def getFeatureType[T <: FeatureType](f: TransientFeature)(implicit conv: FeatureTypeSparkConverter[T]): T =
      conv.fromSpark(getAny(f.name))

  }

}
