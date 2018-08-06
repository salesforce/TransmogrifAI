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

import org.apache.spark.sql.types._

import scala.collection.mutable.{Map => MMap}
import shapeless._

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
              s"Failed to merge metadata for key $key due to incompatible value types '$av' and '$bv'"
            )
          }
        res += key -> resVal
      })

      new MetadataWrapper(res.toMap).metadata
    }

    /**
     * Equals method that will recursively check Metadata objects that contain Metadata values or values that are
     * Array[Metadata]
     *
     * @param that Other metadata object to compare to
     * @return
     */
    def deepEquals(that: Metadata): Boolean = {
      val map1 = this.wrapped.underlyingMap
      val map2 = that.wrapped.underlyingMap

      if (map1.size == map2.size) {
        map1.keysIterator.forall { key =>
          map2.get(key) match {
            case Some(otherValue) =>
              val ourValue = map1(key)
              (ourValue, otherValue) match {
                // Note: Spark will treat any empty Array as an Array[Long], so == will not work here if it thinks
                // one array is an empty Array[Long] and the other is an empty Array[Metadata]
                case (v0: Array[_], v1: Array[_]) => v0.sameElements(v1)
                case (v0: Metadata, v1: Metadata) => v0.deepEquals(v1)
                case (v0, v1) => v0 == v1
              }
            case None => false
          }
        }
      }
      else false
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

  private val booleanSeq = TypeCase[Seq[Boolean]]
  private val longSeq = TypeCase[Seq[Long]]
  private val intSeq = TypeCase[Seq[Int]]
  private val doubleSeq = TypeCase[Seq[Double]]
  private val stringSeq = TypeCase[Seq[String]]

  /**
   * Enrichment functions for Maps
   * @param theMap Map[String, Any]
   */
  implicit class RichMap(val theMap: Map[String, Any]) extends AnyVal {

    def toMetadata: Metadata = {
      val builder = new MetadataBuilder()
      def unsupported(k: String) = throw new RuntimeException(s"Key '$k' has unsupported value type")
      def putCollection(key: String, seq: Seq[Any]): MetadataBuilder = seq match {
        case booleanSeq(v) => builder.putBooleanArray(key, v.toArray)
        case intSeq(v) => builder.putLongArray(key, v.map(_.toLong).toArray)
        case longSeq(v) => builder.putLongArray(key, v.toArray)
        case doubleSeq(v) => builder.putDoubleArray(key, v.toArray)
        case stringSeq(v) => builder.putStringArray(key, v.toArray)
        case _ => unsupported(key)
      }
      theMap.foldLeft(builder) {
        case (m, (k, v: Boolean)) => m.putBoolean(k, v)
        case (m, (k, v: Double)) => m.putDouble(k, v)
        case (m, (k, v: Long)) => m.putLong(k, v)
        case (m, (k, v: String)) => m.putString(k, v)
        case (m, (k, v: Seq[_])) => putCollection(k, v)
        case (m, (k, v: Array[_])) => putCollection(k, v)
        case (m, (k, v: Map[_, _])) => m.putMetadata(k, v.map { case (k, v) => k.toString -> v }.toMetadata)
        case (_, (k, _)) => unsupported(k)
      }.build()
    }
  }

}
