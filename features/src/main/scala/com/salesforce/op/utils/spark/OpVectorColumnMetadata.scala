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

import com.salesforce.op.features.types.{FeatureType, OPMap}
import com.salesforce.op.utils.json.JsonLike
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import com.salesforce.op.utils.spark.RichMetadata.{RichMetadata => RichMeta}

import scala.reflect.runtime.universe._


/**
 * Represents the metadata a column in a vector.
 *
 * Because we expect every vector column to have been produced by some vectorization process, we provide the
 * name of the feature that led to this column.
 *
 * Also note that each column's indicator value should be unique, meaning that they represent mutually exclusive values.
 * The output of a hashing vectorizer, for instance, does not produce mutually exclusive values.
 *
 * @param parentFeatureName The name of the parent feature(s) for the column. Usually a column has one parent feature,
 *                          but can have many (eg. in the case of multiple Text columns being vectorized using a
 *                          shared hash space)
 * @param parentFeatureType The type of the parent feature(s) for the column
 * @param grouping          The name of the group an column belongs to (usually the parent feature, but in the case
 *                          of Maps, this is the keys). Every other column in the same
 *                          vector that has this grouping should be mutually exclusive to this one. If
 *                          there is no grouping then this field is None
 * @param indicatorValue    A name for an binary indicator value (null indicator or result of a pivot or whatever
 *                          that value is), otherwise [[None]] eg this is none when the column is from a numeric group
 *                          that is not pivoted
 * @param descriptorValue   A name for a value that is continuous (not a binary indicator) eg for geolocation (lat, lon,
 *                          accuracy) or for dates that have been converted to a circular representation the time
 *                          window and x or y coordinate, otherwise [[None]]
 * @param index             Index of the vector this info is associated with (this is updated when
 *                          OpVectorColumnMetadata is passed into [[OpVectorMetadata]]
 */
case class OpVectorColumnMetadata // TODO make separate case classes extending this for categorical and numeric
(
  parentFeatureName: Seq[String],
  parentFeatureType: Seq[String],
  grouping: Option[String],
  indicatorValue: Option[String] = None,
  descriptorValue: Option[String] = None,
  index: Int = 0
) extends JsonLike {

  assert(parentFeatureName.nonEmpty, "must provide parent feature name")
  assert(parentFeatureType.nonEmpty, "must provide parent type name")
  assert(parentFeatureName.length == parentFeatureType.length,
    s"must provide both type and name for every parent feature," +
      s" names: $parentFeatureName and types: $parentFeatureType do not have the same length")
  assert(indicatorValue.isEmpty || descriptorValue.isEmpty, "cannot have both indicatorValue and descriptorValue")

  /**
   * Convert this column into Spark metadata.
   *
   * @param ind Indexes of vector columns that match this OpVectorColumnMetadata
   * @return column Spark metadata
   */
  def toMetadata(ind: Array[Int]): Metadata = {
    val builder = new MetadataBuilder()
      .putStringArray(OpVectorColumnMetadata.ParentFeatureKey, parentFeatureName.toArray)
      .putStringArray(OpVectorColumnMetadata.ParentFeatureTypeKey, parentFeatureType.toArray)
      .putLongArray(OpVectorColumnMetadata.IndicesKey, ind.map(_.toLong))

    grouping.foreach(builder.putString(OpVectorColumnMetadata.GroupingKey, _))
    indicatorValue.foreach(builder.putString(OpVectorColumnMetadata.IndicatorValueKey, _))
    descriptorValue.foreach(builder.putString(OpVectorColumnMetadata.DescriptorValueKey, _))
    builder.build()
  }

  /**
   * Is this column corresponds to a null-encoded categorical (maybe also other types - investigating!)
   * @return true if this column corresponds to a null-encoded categorical (maybe also other types - investigating!)
   */
  def isNullIndicator: Boolean = indicatorValue.contains(OpVectorColumnMetadata.NullString)

  /**
   * Convert this column into Spark metadata.
   *
   * @return column Spark metadata
   */
  def toMetadata(): Metadata = toMetadata(Array(index))

  /**
   * Make unique name for this column
   * @return String name for this column
   */
  def makeColName(): String =
    s"${parentFeatureName.mkString("_")}${grouping.map("_" + _).getOrElse("")}" +
      s"${indicatorValue.map("_" + _).getOrElse("")}${descriptorValue.map("_" + _).getOrElse("")}_$index"

  /**
   * Does column have parent features of specified feature type O
   */
  def hasParentOfType[O <: FeatureType](implicit tt: TypeTag[O]): Boolean =
    parentFeatureType.exists { parentTypeName =>
      FeatureType.featureTypeTag(parentTypeName).tpe =:= tt.tpe
    }

  /**
   * Does column have parent features of which are subtypes of feature type O
   */
  def hasParentOfSubType[O <: FeatureType](implicit tt: TypeTag[O]): Boolean =
    parentFeatureType.exists { parentTypeName =>
      FeatureType.featureTypeTag(parentTypeName).tpe <:< tt.tpe
    }

  /**
   * Return parent features names with the key (grouping) from any map parents included in name
   * @return Sequence of parent feature names, simple names when features are not maps, names plus keys
   *         for columns with map parent features
   */
  def parentNamesWithMapKeys(): Seq[String] =
    if (hasParentOfSubType[OPMap[_]]) parentFeatureName.map(p => grouping.map(p + "_" + _).getOrElse(p))
    else parentFeatureName

}

object OpVectorColumnMetadata {
  val ParentFeatureKey = "parent_feature"
  val ParentFeatureTypeKey = "parent_feature_type"
  val GroupingKey = "grouping"
  val IndicatorValueKey = "indicator_value"
  val DescriptorValueKey = "descriptor_value"
  val IndicesKey = "indices"
  val NullString = "NullIndicatorValue"

  /**
   * Alternate constructor for OpVectorColumnMetadata cannot be in class because causes serialization issues
   * @param parentFeatureName The name of the parent feature(s) for the column. Usually a column has one parent feature,
   *                          but can have many (eg. in the case of multiple Text columns being vectorized using a
   *                          shared hash space)
   * @param parentFeatureType The type of the parent feature(s) for the column
   * @param grouping          The name of the group a column belongs to (usually the parent feature, but in the case
   *                          of MapVectorizers, this includes keys in maps too). Every other vector column in the
   *                          same vector that has this same indicator group should be mutually exclusive to this one.
   * @param indicatorValue    An indicator for a value for a binary column (null indicator or result of a pivot or
   *                          whatever that value is), otherwise [[None]]
   * @param descriptorValue   A name for a value that is continuous (not a binary indicator) eg for geolocation (lat,
   *                          lon, accuracy) or for dates that have been converted to a circular representation the time
   *                          window and x or y coordinate, otherwise [[None]]
   * @return new OpVectorColumnMetadata
   */
  def apply(parentFeatureName: Seq[String],
    parentFeatureType: Seq[String],
    grouping: Seq[String],
    indicatorValue: Option[String],
    descriptorValue: Option[String]
  ): OpVectorColumnMetadata = OpVectorColumnMetadata(parentFeatureName, parentFeatureType,
    if (grouping.nonEmpty) Option(grouping.mkString("_")) else None,
    indicatorValue, descriptorValue, 0)

  /**
   * Build an [[OpVectorColumnMetadata]] from Spark metadata representing a column.
   *
   * @param meta The metadata to build column from
   * @return The built [[OpVectorColumnMetadata]]
   */
  def fromMetadata(meta: Metadata): Array[OpVectorColumnMetadata] = {
    val wrp = RichMeta(meta).wrapped
    val ind = wrp.getArray[Long](IndicesKey)
    val info = OpVectorColumnMetadata(
      parentFeatureName = wrp.getArray[String](ParentFeatureKey),
      parentFeatureType = wrp.getArray[String](ParentFeatureTypeKey),
      grouping = if (wrp.contains(GroupingKey)) Option(wrp.get[String](GroupingKey)) else None,
      indicatorValue = if (wrp.contains(IndicatorValueKey)) Option(wrp.get[String](IndicatorValueKey)) else None,
      descriptorValue = if (wrp.contains(DescriptorValueKey)) Option(wrp.get[String](DescriptorValueKey)) else None
    )
    ind.map(i => info.copy(index = i.toInt))
  }
}
