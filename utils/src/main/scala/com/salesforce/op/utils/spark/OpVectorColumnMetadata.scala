/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.utils.json.JsonLike
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import com.salesforce.op.utils.spark.RichMetadata.{RichMetadata => RichMeta}


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
 * @param indicatorGroup    The name of the group an indicator belongs to (usually the parent feature, but in the case
 *                          of Maps, this is the keys). Every other column in the same
 *                          vector that has this indicator group should be mutually exclusive to this one. If
 *                          there is no grouping then this field is None
 * @param indicatorValue    An indicator for a value (null indicator or result of a pivot or whatever that value is),
 *                          otherwise [[None]] eg this is none when the column is from a numeric group that is not
 *                          pivoted
 * @param index             Index of the vector this info is associated with (this is updated when
 *                          OpVectorColumnMetadata is passed into [[OpVectorMetadata]]
 */
case class OpVectorColumnMetadata
(
  parentFeatureName: Seq[String],
  parentFeatureType: Seq[String],
  indicatorGroup: Option[String],
  indicatorValue: Option[String],
  index: Int = 0
) extends JsonLike {

  assert(parentFeatureName.nonEmpty, "must provide parent feature name")
  assert(parentFeatureType.nonEmpty, "must provide parent type name")
  assert(parentFeatureName.length == parentFeatureType.length,
    s"must provide both type and name for every parent feature," +
      s" names: $parentFeatureName and types: $parentFeatureType do not have the same length")

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

    indicatorGroup.foreach(builder.putString(OpVectorColumnMetadata.IndicatorGroupKey, _))
    indicatorValue.foreach(builder.putString(OpVectorColumnMetadata.IndicatorValueKey, _))
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
    s"${parentFeatureName.mkString("_")}${indicatorGroup.map("_" + _).getOrElse("")}" +
      s"${indicatorValue.map("_" + _).getOrElse("")}_$index"

  /**
   * Does column have parent features that are maps
   * @return boolean indicating whether parent feature type sequence contains Map types
   */
  def hasMapParent(): Boolean = {
    // TODO: move this class to `features` or `core` sub project to avoid mentioning types as strings
    hasParentOfType("Map") || hasParentOfType("Prediction")
  }

  /**
   * Does column have parent features of specified feature type
   * @return boolean indicating whether parent feature type sequence contains type name
   */
  def hasParentOfType(typeName: String): Boolean = parentFeatureType.exists(_.contains(typeName))

  /**
   * Return parent features names with the key (indicatorGroup) from any map parents included in name
   * @return Sequence of parent feature names, simple names when features are not maps, names plus keys
   *         for columns with map parent features
   */
  def parentNamesWithMapKeys(): Seq[String] =
    if (hasMapParent()) parentFeatureName.map(p => indicatorGroup.map(p + "_" + _).getOrElse(p))
    else parentFeatureName

}

object OpVectorColumnMetadata {
  val ParentFeatureKey = "parent_feature"
  val ParentFeatureTypeKey = "parent_feature_type"
  val IndicatorGroupKey = "indicator_group"
  val IndicatorValueKey = "indicator_value"
  val IndicesKey = "indices"
  val NullString = "NullIndicatorValue"

  /**
   * Alternate constructor for OpVectorColumnMetadata cannot be in class because causes serialization issues
   * @param parentFeatureName The name of the parent feature(s) for the column. Usually a column has one parent feature,
   *                          but can have many (eg. in the case of multiple Text columns being vectorized using a
   *                          shared hash space)
   * @param parentFeatureType The type of the parent feature(s) for the column
   * @param indicatorGroup    The name of the group an indicator belongs to (usually the parent feature, but in the case
   *                          of TextMapVectorizer, this includes keys in maps too). Every other vector column in the
   *                          same vector that has this same indicator group should be mutually exclusive to this one.
   *                          If this is not an indicator, or it corresponds to a null indicator, then field is None
   * @param indicatorValue    An indicator for a value (null indicator or result of a pivot or whatever that value is),
   *                          otherwise [[None]]
   * @return new OpVectorColumnMetadata
   */
  def apply(parentFeatureName: Seq[String],
    parentFeatureType: Seq[String],
    indicatorGroup: Seq[String],
    indicatorValue: Option[String]
  ): OpVectorColumnMetadata = OpVectorColumnMetadata(parentFeatureName, parentFeatureType,
    if (indicatorGroup.nonEmpty) Option(indicatorGroup.mkString("_")) else None,
    indicatorValue, 0)

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
      indicatorGroup = if (wrp.contains(IndicatorGroupKey)) Option(wrp.get[String](IndicatorGroupKey)) else None,
      indicatorValue = if (wrp.contains(IndicatorValueKey)) Option(wrp.get[String](IndicatorValueKey)) else None
    )
    ind.map(i => info.copy(index = i.toInt))
  }
}
