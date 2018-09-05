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

import com.salesforce.op.FeatureHistory
import com.salesforce.op.features.types.{Binary, BinaryMap, Text, TextArea, TextAreaMap, TextMap}
import org.apache.spark.ml.attribute.{AttributeGroup, BinaryAttribute, NominalAttribute, NumericAttribute}
import org.apache.spark.ml.linalg.SQLDataTypes._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField}

/**
 * Represents a metadata wrapper that includes parent feature information.
 *
 * The metadata includes a columns field that describes the columns in the vector.
 *
 * @param name    name of the feature vector
 * @param col     information about each element in the vector
 * @param history history of parent features used to create the vector map is from
 *                OpVectorColumnMetadata.parentFeatureName (String) to FeatureHistory
 */
class OpVectorMetadata private
(
  val name: String,
  col: Array[OpVectorColumnMetadata],
  val history: Map[String, FeatureHistory]
) {

  /**
   * Column metadata with indicies fixed to match order passed in
   */
  val columns: Array[OpVectorColumnMetadata] = col.zipWithIndex.map { case (c, i) => c.copy(index = i) }

  /**
   * Get the number of columns in vectors of this type
   *
   * @return Number of columns as int
   */
  def size: Int = columns.length

  /**
   * Return a new instance of [[OpVectorMetadata]] with the given columns used to update columns with value information
   *
   * @param newColumns New columns as an array of [[OpVectorMetadata]]
   * @return New vector metadata
   */
  def withColumns(
    newColumns: Array[OpVectorColumnMetadata]
  ): OpVectorMetadata = OpVectorMetadata(name, newColumns, history)

  val textTypes = Seq(Text, TextArea, TextAreaMap, TextMap, Binary, BinaryMap).map(_.getClass.toString.dropRight(1))
  /**
   * Serialize to spark metadata
   *
   * @return Spark metadata
   */
  def toMetadata: Metadata = {
    val groupedCol = columns
      .groupBy(c => (c.parentFeatureName, c.parentFeatureType, c.grouping, c.indicatorValue, c.descriptorValue))
    val colData = groupedCol.toSeq
      .map { case (_, g) => g.head -> g.map(_.index) }
    val colMeta = colData.map { case (c, i) =>
      c.toMetadata(i)
    }
    val meta = new MetadataBuilder()
      .putMetadataArray(OpVectorMetadata.ColumnsKey, colMeta.toArray)
      .putMetadata(OpVectorMetadata.HistoryKey, FeatureHistory.toMetadata(history))
      .build()
    val attributes = columns.map { c =>
      if (c.indicatorValue.isDefined || textTypes.exists(c.parentFeatureType.contains)) {
        BinaryAttribute.defaultAttr.withName(c.makeColName()).withIndex(c.index)
      } else {
        NumericAttribute.defaultAttr.withName(c.makeColName()).withIndex(c.index)
      }
    }
    new AttributeGroup(name, attributes).toMetadata(meta)
  }

  /**
   * Serialize to spark metadata inside a StructField
   *
   * @return Spark struct field
   */
  def toStructField(): StructField = {
    StructField(name, VectorType, nullable = false, toMetadata)
  }


  /**
   * Extract the full history for each element of the vector
   *
   * @return Sequence of [[OpVectorColumnHistory]]
   */
  def getColumnHistory(): Seq[OpVectorColumnHistory] = {
    columns.map { c =>
      val hist = c.parentFeatureName.map(pn => history.getOrElse(pn,
        throw new RuntimeException(s"Parent feature name '${pn}' has no associated history")))
      val histComb = hist.head.merge(hist.tail: _*)
      OpVectorColumnHistory(
        columnName = c.makeColName(),
        parentFeatureName = c.parentFeatureName,
        parentFeatureOrigins = histComb.originFeatures,
        parentFeatureStages = histComb.stages,
        parentFeatureType = c.parentFeatureType,
        grouping = c.grouping,
        indicatorValue = c.indicatorValue,
        descriptorValue = c.descriptorValue,
        index = c.index
      )
    }
  }

  /**
   * Get index of the given [[OpVectorColumnMetadata]], or throw an error if it isn't in this vector metadata or
   * if multiple instances of it are in this metadata
   *
   *
   * @param column The column to check
   * @return The index of the column
   * @throws IllegalArgumentException if the column does not appear exactly once in this vector
   */
  def index(column: OpVectorColumnMetadata): Int = {
    val matchingCols = columns.view.zipWithIndex.filter(_._1 == column)
    if (matchingCols.isEmpty) {
      throw new IllegalArgumentException(s"No instance of $column found in $this")
    } else if (matchingCols.size >= 2) {
      val indices = matchingCols.map(_._2).mkString(", ")
      throw new IllegalArgumentException(s"Multiple instances of $column found in $this at indices $indices")
    } else {
      matchingCols.head._2
    }
  }

  // have to override to get better Array equality
  override def equals(obj: Any): Boolean =
    obj match {
      case o: OpVectorMetadata
        if o.name == name && o.columns.toSeq == columns.toSeq && history == o.history => true
      case _ => false
    }

  // have to override to support overridden .equals
  override def hashCode(): Int = 37 * columns.toSeq.hashCode()

  override def toString: String =
    s"${this.getClass.getSimpleName}($name,${columns.mkString("Array(", ",", ")")},$history)"

}

object OpVectorMetadata {

  import com.salesforce.op.utils.spark.RichMetadata._

  val ColumnsKey = "vector_columns"
  val HistoryKey = "vector_history"

  /**
   * Construct an [[OpVectorMetadata]] from a [[StructField]], assuming that [[ColumnsKey]] is present and conforms
   * to an array of [[OpVectorColumnMetadata]]
   *
   * @param field The struct field to build from
   * @return The constructed vector metadata
   */
  def apply(field: StructField): OpVectorMetadata = {
    val wrapped = field.metadata.wrapped

    val columns: Array[OpVectorColumnMetadata] =
      wrapped.getArray[Metadata](ColumnsKey).flatMap(OpVectorColumnMetadata.fromMetadata).sortBy(_.index)

    val history =
      if (wrapped.underlyingMap(HistoryKey).asInstanceOf[Metadata].isEmpty) Map.empty[String, FeatureHistory]
      else FeatureHistory.fromMetadataMap(field.metadata.getMetadata(HistoryKey))

    new OpVectorMetadata(field.name, columns, history)
  }


  /**
   * Construct an [[OpVectorMetadata]] from a string representing its name, and an array of [[OpVectorColumnMetadata]]
   * representing its columns.
   *
   * @param name    The name of the column the metadata represents
   * @param columns The columns within the vectors
   * @param history The history of the parent features
   * @return The constructed vector metadata
   */
  def apply(
    name: String,
    columns: Array[OpVectorColumnMetadata],
    history: Map[String, FeatureHistory]
  ): OpVectorMetadata = {
    new OpVectorMetadata(name, columns, history)
  }

  /**
   * Construct an [[OpVectorMetadata]] from its name and a [[Metadata]], assuming that [[ColumnsKey]] and
   * [[HistoryKey]] are present and conforms to an array of [[OpVectorColumnMetadata]]
   * and [[ Map[String, FeatureHistory] ]]
   *
   * @param name     The name of the vector metadata
   * @param metadata The metadata to build from
   * @return The constructed vector metadata
   */
  def apply(name: String, metadata: Metadata): OpVectorMetadata = {
    val field = StructField(name, dataType = VectorType, nullable = false, metadata = metadata)
    apply(field)
  }

  /**
   * Flatten a Seq[OpVectorMetadata] into one OpVectorMetadata by concatenating the vectors
   *
   * @param outputName Name of the output flattened metadata
   * @param vectors    List of vector metadata to flatten
   * @return Flattened metadata
   */
  def flatten(outputName: String, vectors: Seq[OpVectorMetadata]): OpVectorMetadata = {
    val allColumns = vectors.flatMap(_.columns).toArray
    val allHist = vectors.flatMap(_.history).toMap
    new OpVectorMetadata(outputName, allColumns, allHist)
  }

}
