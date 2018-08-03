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

import org.apache.spark.sql.types.{ArrayType, DataType, MapType, StructType}

object RichDataType {

  /**
   * Enrichment functions for DataType
   *
   * @param dt DataType
   */
  implicit class RichDataType[T <: DataType](val dt: T) extends AnyVal {

    /**
     * Compares to another type, ignoring compatible nullability of ArrayType, MapType, StructType.
     *
     * @param that Datatype
     * @return true if types are equal (ignoring nullability), false otherwise
     */
    def equalsIgnoreNullability(that: DataType): Boolean = (dt, that) match {
      case (ArrayType(lt, _), ArrayType(rt, _)) => lt.equalsIgnoreNullability(rt)
      case (MapType(lk, lv, _), MapType(rk, rv, _)) =>
        lk.equalsIgnoreNullability(rk) && lv.equalsIgnoreNullability(rv)
      case (StructType(lf), StructType(rf)) =>
        lf.length == rf.length &&
          lf.zip(rf).forall { case (l, r) => l.name == r.name && l.dataType.equalsIgnoreNullability(r.dataType) }
      case (fromDataType, toDataType) => fromDataType == toDataType
    }

  }

}
