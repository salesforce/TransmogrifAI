/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
