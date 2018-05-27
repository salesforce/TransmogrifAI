/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
