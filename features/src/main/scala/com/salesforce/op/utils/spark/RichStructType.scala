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

object RichStructType {

  /**
   * Enrichment functions for StructType (i.e. Dataset / Dataframe schema)
   *
   * @param st StructType
   */
  implicit class RichStructType(val st: StructType) extends AnyVal {

    /**
     * Find similar field names in struct schema schema
     *
     * @param query      approximate field name query
     * @param ignoreCase should ignore case when comparing field names
     * @return field names approximately matching the requested field name
     */
    def findFields(query: String, ignoreCase: Boolean = true): Seq[StructField] = {
      def preProcess(s: String): String = {
        val noDigits = s.replaceAll("\\d", "")
        if (ignoreCase) noDigits.toLowerCase else noDigits
      }
      val q = preProcess(query)
      for {
        field <- st.fields
        f = preProcess(field.name)
        if query == field.name || f == q || f.contains(q)
      } yield field
    }

    /**
     * Find a similar field name in struct schema schema
     *
     * @param query      approximate field name query
     * @param ignoreCase should ignore case when comparing field names
     * @throws IllegalArgumentException if no unique or multiple matching field were found
     * @return field names approximately matching the requested field name
     */
    def findField(query: String, ignoreCase: Boolean = true): StructField = {
      findFields(query, ignoreCase).toList match {
        case Nil => throw new IllegalArgumentException(s"No unique field found matching '$query'")
        case field :: Nil => field
        case fields => throw new IllegalArgumentException(
          s"Multiple fields matching '$query': " + fields.map(_.name).mkString("'", "','", "'")
        )
      }
    }

    /**
     * Create [[OpVectorMetadata]] from a field name
     *
     * @param fieldName field name
     * @throws IllegalArgumentException if a field with the given name does not exist
     * @return [[OpVectorMetadata]] for a field
     */
    def toOpVectorMetadata(fieldName: String): OpVectorMetadata =
      OpVectorMetadata(fieldName, st(fieldName).metadata)

  }

}
