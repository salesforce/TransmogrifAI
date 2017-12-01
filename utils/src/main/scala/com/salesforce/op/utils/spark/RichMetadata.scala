/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import org.apache.spark.sql.types._

import scala.collection.mutable.{Map => MMap}

object RichMetadata {

  val SummaryKey = "summary"

  /**
   * Enrichment functions for Metadata
   *
   * @param metadata Metadata
   */
  implicit class RichMetadata(val metadata: Metadata) extends AnyVal {

    /**
     * Returns a wrapped metadata which allows access to private members of metadata
     *
     * @return wrapped metadata
     */
    def wrapped: MetadataWrapper = new MetadataWrapper(metadata)

    /**
     * Get underlying untyped Map[String, Any]
     *
     * @return underlying untyped Map[String, Any]
     */
    def underlyingMap: Map[String, Any] = wrapped.underlyingMap

    /**
     * Tests whether the map is empty.
     *
     * @return `true` if the map does not contain any key/value binding, `false` otherwise.
     */
    def isEmpty: Boolean = underlyingMap.isEmpty

    /**
     * Converts to its pretty json representation
     *
     * @return pretty json string
     */
    def prettyJson: String = wrapped.prettyJson

    /**
     * Deep merge this Metadata with that one
     *
     * @param that Metadata
     * @return merged Metadata
     */
    def deepMerge(that: Metadata): Metadata = {
      val a = new MetadataWrapper(metadata)
      val b = new MetadataWrapper(that)
      val keys = a.underlyingMap.keySet ++ b.underlyingMap.keySet
      val res = MMap.empty[String, Any]

      keys.foreach(key => {
        val resVal =
          (a.getAny(key), b.getAny(key)) match {
            case (None, None) => // nothing to do
            case (None, Some(v)) => v
            case (Some(v), None) => v
            case (Some(av: Array[Long]), Some(bv: Array[Long])) => av ++ bv
            case (Some(av: Array[Double]), Some(bv: Array[Double])) => av ++ bv
            case (Some(av: Array[Boolean]), Some(bv: Array[Boolean])) => av ++ bv
            case (Some(av: Array[String]), Some(bv: Array[String])) => av ++ bv
            case (Some(av: Array[Metadata]), Some(bv: Array[Metadata])) => av ++ bv
            case (Some(av: Long), Some(bv: Long)) => av + bv
            case (Some(av: Double), Some(bv: Double)) => av + bv
            case (Some(av: Boolean), Some(bv: Boolean)) => av || bv
            case (Some(av: String), Some(bv: String)) => av + bv
            case (Some(av: Metadata), Some(bv: Metadata)) => av.deepMerge(bv)
            case (Some(av), Some(bv)) => throw new RuntimeException(
              s"Failed to merge metadatas for key $key due to incompatible value types '$av' and '$bv'"
            )
          }
        res += key -> resVal
      })

      new MetadataWrapper(res.toMap).metadata
    }

    /**
     * Add summary metadata to an existing metadata instance
     *
     * @param summary Metadata containing any summary information from estimator
     * @return a new combined instance of metadata
     */
    def withSummaryMetadata(summary: Metadata): Metadata = {
      new MetadataBuilder().withMetadata(metadata).putMetadata(SummaryKey, summary).build()
    }

    /**
     * Turn an this metadata into summary metadata by putting it behind the summary key
     *
     * @return a new metadata instance
     */
    def toSummaryMetadata(): Metadata = {
      new MetadataBuilder().putMetadata(SummaryKey, metadata).build()
    }

    /**
     * Get summary metadata
     *
     * @return metadata under summary key
     */
    def getSummaryMetadata(): Metadata = {
      metadata.getMetadata(SummaryKey)
    }

    /**
     * Checks if metadata contains summary
     *
     * @return boolean value - true if summary exists false if not
     */
    def containsSummaryMetadata(): Boolean = {
      metadata.contains(SummaryKey)
    }
  }

  /**
   * Enrichment functions for Maps
   * @param theMap Map[String, Any]
   */
  implicit class RichMap(val theMap: Map[String, Any]) extends AnyVal {

    def toMetadata: Metadata = theMap.foldLeft(new MetadataBuilder()) {
      case (m, (k, v: Boolean)) => m.putBoolean(k, v)
      case (m, (k, v: Double)) => m.putDouble(k, v)
      case (m, (k, v: Long)) => m.putLong(k, v)
      case (m, (k, v: String)) => m.putString(k, v)
      case (m, (k, v: Array[Boolean])) => m.putBooleanArray(k, v)
      case (m, (k, v: Array[Double])) => m.putDoubleArray(k, v)
      case (m, (k, v: Array[Long])) => m.putLongArray(k, v)
      case (m, (k, v: Array[String])) => m.putStringArray(k, v)
      case (_, (k, v)) => throw new RuntimeException(s"Key '$k' has unsupported value type")
    }.build()

  }

}
