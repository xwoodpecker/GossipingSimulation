# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gossip.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cgossip.proto\x12\x06gossip\"\x0e\n\x0cResetRequest\"\x0f\n\rResetResponse\"\x0f\n\rGossipRequest\"\x10\n\x0eGossipResponse\"\x10\n\x0eHistoryRequest\"<\n\x0fHistoryResponse\x12)\n\rvalue_entries\x18\x01 \x03(\x0b\x32\x12.gossip.ValueEntry\"3\n\nValueEntry\x12\x16\n\x0eparticipations\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05\"\x15\n\x13\x43urrentValueRequest\"%\n\x14\x43urrentValueResponse\x12\r\n\x05value\x18\x01 \x01(\x05\"\x18\n\x16StopApplicationRequest\"\x19\n\x17StopApplicationResponse2\xdc\x02\n\x06Gossip\x12\x36\n\x05Reset\x12\x14.gossip.ResetRequest\x1a\x15.gossip.ResetResponse\"\x00\x12\x39\n\x06Gossip\x12\x15.gossip.GossipRequest\x1a\x16.gossip.GossipResponse\"\x00\x12<\n\x07History\x12\x16.gossip.HistoryRequest\x1a\x17.gossip.HistoryResponse\"\x00\x12K\n\x0c\x43urrentValue\x12\x1b.gossip.CurrentValueRequest\x1a\x1c.gossip.CurrentValueResponse\"\x00\x12T\n\x0fStopApplication\x12\x1e.gossip.StopApplicationRequest\x1a\x1f.gossip.StopApplicationResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'gossip_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RESETREQUEST._serialized_start=24
  _RESETREQUEST._serialized_end=38
  _RESETRESPONSE._serialized_start=40
  _RESETRESPONSE._serialized_end=55
  _GOSSIPREQUEST._serialized_start=57
  _GOSSIPREQUEST._serialized_end=72
  _GOSSIPRESPONSE._serialized_start=74
  _GOSSIPRESPONSE._serialized_end=90
  _HISTORYREQUEST._serialized_start=92
  _HISTORYREQUEST._serialized_end=108
  _HISTORYRESPONSE._serialized_start=110
  _HISTORYRESPONSE._serialized_end=170
  _VALUEENTRY._serialized_start=172
  _VALUEENTRY._serialized_end=223
  _CURRENTVALUEREQUEST._serialized_start=225
  _CURRENTVALUEREQUEST._serialized_end=246
  _CURRENTVALUERESPONSE._serialized_start=248
  _CURRENTVALUERESPONSE._serialized_end=285
  _STOPAPPLICATIONREQUEST._serialized_start=287
  _STOPAPPLICATIONREQUEST._serialized_end=311
  _STOPAPPLICATIONRESPONSE._serialized_start=313
  _STOPAPPLICATIONRESPONSE._serialized_end=338
  _GOSSIP._serialized_start=341
  _GOSSIP._serialized_end=689
# @@protoc_insertion_point(module_scope)
