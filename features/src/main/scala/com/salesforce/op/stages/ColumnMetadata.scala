package com.salesforce.op.stages

import org.apache.spark.sql.types.{Metadata, StructType}

object ColumnMetadata {
  /** An implicit class to insert column metadata into a spark schema (StructType) */
  implicit class SchemaWithColumnMetadata(schema: StructType) {
    /** inserts column metadata into a spark schema from a metadata object. If there's no metadata for given column,
     * nothing is inserted. */
    def insertColumnMetadata(elems: (String, Metadata)*): StructType = {
      val fieldsWithMetadata = schema.map { case field =>
        elems.toMap.get(field.name) match {
          case Some(metadata: Metadata) => field.copy(metadata = metadata)
          case _ => field
        }
      }
      StructType(fieldsWithMetadata)
    }
  }
}
