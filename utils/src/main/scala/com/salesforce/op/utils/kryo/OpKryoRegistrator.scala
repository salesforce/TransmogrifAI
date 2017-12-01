/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.kryo


import com.esotericsoftware.kryo.{Kryo, Registration}
import com.twitter.chill.avro.AvroSerializer
import org.apache.avro.generic.GenericData
import org.apache.avro.specific.SpecificRecordBase
import org.apache.spark.serializer.KryoRegistrator

import scala.reflect._


class OpKryoRegistrator extends KryoRegistrator {

  protected def doAvroRegistration[T <: SpecificRecordBase : ClassTag](kryo: Kryo): Registration =
    kryo.register(classTag[T].runtimeClass, AvroSerializer.SpecificRecordBinarySerializer[T])

  protected def doClassRegistration(kryo: Kryo)(seqPC: Class[_]*): Unit =
    seqPC.foreach { pC =>
      kryo.register(pC)
      // also register arrays of that class
      val arrayType = java.lang.reflect.Array.newInstance(pC, 0).getClass
      kryo.register(arrayType)
    }

  final override def registerClasses(kryo: Kryo): Unit = {
    doClassRegistration(kryo)(
      classOf[org.apache.avro.generic.GenericData],
      scala.collection.immutable.Map.empty[Any, Any].getClass
    )
    doClassRegistration(kryo)(
      OpKryoClasses.ArraysOfPrimitives: _*
    )
    // Avro generic-data array deserialization fails - hence providing workaround
    kryo.register(
      classOf[GenericData.Array[_]],
      new GenericJavaCollectionSerializer(classOf[java.util.ArrayList[_]])
    )
    registerCustomClasses(kryo)
  }

  /**
   * Override this method to register custom types
   *
   * @param kryo
   */
  protected def registerCustomClasses(kryo: Kryo): Unit = {}

}

private[op] case object OpKryoClasses {

  lazy val ArraysOfPrimitives: Seq[Class[_]] = Seq(
    Class.forName("[Z") /* boolean[] */,
    Class.forName("[B") /* byte[] */,
    Class.forName("[C") /* char[] */,
    Class.forName("[D") /* double[] */,
    Class.forName("[F") /* float[] */,
    Class.forName("[I") /* int[] */,
    Class.forName("[J") /* long[] */,
    Class.forName("[S") /* short[] */
  )

}
