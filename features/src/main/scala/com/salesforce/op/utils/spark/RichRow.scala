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
     *
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
    def getOption[T](fieldName: String): Option[T] = getOptionAny(fieldName) collect { case t: T@unchecked => t }

    /**
     * Returns the value at position i. If the value is null, None is returned.
     */
    def getOption[T](i: Integer): Option[T] = getOptionAny(i) collect { case t: T@unchecked => t }

    /**
     * Returns the value of a given feature casted into the feature type
     *
     * @throws UnsupportedOperationException when schema is not defined.
     * @throws IllegalArgumentException      when fieldName do not exist.
     * @throws ClassCastException            when data type does not match.
     */
    def getFeatureType[T <: FeatureType](f: FeatureLike[T])(implicit conv: FeatureTypeSparkConverter[T]): T =
      conv.fromSpark(getAny(f.name))

    /**
     * Returns the value of a given feature casted into the feature type from the transient feature and the
     * weak type tag of features
     *
     * @throws UnsupportedOperationException when schema is not defined.
     * @throws IllegalArgumentException      when fieldName do not exist.
     * @throws ClassCastException            when data type does not match.
     */
    def getFeatureType[T <: FeatureType](f: TransientFeature)(implicit conv: FeatureTypeSparkConverter[T]): T =
      conv.fromSpark(getAny(f.name))

    /**
     * Converts row to a [[collection.immutable.Map]]
     *
     * @return a [[collection.immutable.Map]] with all row contents
     */
    def toMutableMap: collection.mutable.Map[String, Any] = {
      val res = collection.mutable.Map.empty[String, Any]
      val fields = row.schema.fields
      for {i <- 0 until row.size} res.put(fields(i).name, row(i))
      res
    }

    /**
     * Converts row to a [[collection.mutable.Map]]
     *
     * @return a [[collection.mutable.Map]] with all row contents
     */
    def toMap: Map[String, Any] = toMutableMap.toMap

  }

}
