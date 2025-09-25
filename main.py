"""
FastAPI Backend for Lease Request Management System
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 3002
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text, ForeignKey, Date, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import os
from dotenv import load_dotenv
import uuid
import json

import uvicorn

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ========================================
# DATABASE MODELS
# ========================================

class LeaseRequestDB(Base):
    __tablename__ = "lease_requests"
    
    id = Column(String(50), primary_key=True, index=True)
    property_id = Column(String(50), nullable=False)
    tenant_name = Column(String(255), nullable=False)
    tenant_abn = Column(String(20))
    tenant_acn = Column(String(20))
    property_address = Column(Text, nullable=False)
    lease_term = Column(Integer, nullable=False)
    commencement_date = Column(Date, nullable=False)
    rent_amount = Column(DECIMAL(10, 2), nullable=False)
    security_deposit = Column(DECIMAL(10, 2), nullable=False)
    status = Column(String(50), nullable=False)
    requestor_email = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    documents = relationship("LeaseDocumentDB", back_populates="lease_request")
    audit_entries = relationship("AuditEntryDB", back_populates="lease_request")
    workflow_steps = relationship("WorkflowStepDB", back_populates="lease_request")

class PropertyDB(Base):
    __tablename__ = "properties"
    
    id = Column(String(50), primary_key=True, index=True)
    address = Column(Text, nullable=False)
    unit_number = Column(String(100))
    property_type = Column(String(50), nullable=False)
    usage_type = Column(String(100))
    available_area = Column(Integer)
    sap_id = Column(String(100))
    is_available = Column(Boolean, default=True)

class BusinessPartnerDB(Base):
    __tablename__ = "business_partners"
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    abn = Column(String(20))
    acn = Column(String(20))
    address = Column(Text)
    phone = Column(String(50))
    email = Column(String(255))
    sap_id = Column(String(100))
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

class WorkflowStepDB(Base):
    __tablename__ = "workflow_steps"
    
    id = Column(String(50), primary_key=True, index=True)
    lease_request_id = Column(String(50), ForeignKey("lease_requests.id"))
    step_number = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), nullable=False, default='pending')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    assigned_to = Column(String(255))
    notes = Column(Text)
    confidence_score = Column(DECIMAL(3, 2))
    requires_review = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship
    lease_request = relationship("LeaseRequestDB", back_populates="workflow_steps")

class LeaseDocumentDB(Base):
    __tablename__ = "lease_documents"
    
    id = Column(String(50), primary_key=True, index=True)
    lease_request_id = Column(String(50), ForeignKey("lease_requests.id"))
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    url = Column(Text)
    size = Column(Integer)
    confidence_score = Column(DECIMAL(3, 2))
    uploaded_at = Column(DateTime, default=func.now())
    
    # Relationship
    lease_request = relationship("LeaseRequestDB", back_populates="documents")

class AuditEntryDB(Base):
    __tablename__ = "audit_entries"
    
    id = Column(String(50), primary_key=True, index=True)
    lease_request_id = Column(String(50), ForeignKey("lease_requests.id"))
    action = Column(String(255), nullable=False)
    details = Column(Text)
    performed_by = Column(String(255), nullable=False)
    step_number = Column(Integer)
    confidence_score = Column(DECIMAL(3, 2))
    sla_breached = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationship
    lease_request = relationship("LeaseRequestDB", back_populates="audit_entries")

# ========================================
# DATABASE SETUP
# ========================================

# Create tables
Base.metadata.create_all(bind=engine)

# ========================================
# PYDANTIC MODELS
# ========================================
class WorkflowStepResponse(BaseModel):
    id: str
    stepNumber: int  # camelCase for frontend
    name: str
    description: Optional[str] = None
    status: str
    startedAt: Optional[datetime] = None  # camelCase for frontend
    completedAt: Optional[datetime] = None  # camelCase for frontend
    assignedTo: Optional[str] = None  # camelCase for frontend
    notes: Optional[str] = None
    confidenceScore: Optional[float] = None  # camelCase for frontend
    requiresReview: bool = False  # camelCase for frontend
    createdAt: datetime  # camelCase for frontend
    updatedAt: datetime  # camelCase for frontend
    
    class Config:
        from_attributes = True

class WorkflowStepCreate(BaseModel):
    stepNumber: int  # camelCase for frontend
    name: str
    description: Optional[str] = None
    status: str = 'pending'
    assignedTo: Optional[str] = None  # camelCase for frontend
    notes: Optional[str] = None
    confidenceScore: Optional[float] = None  # camelCase for frontend
    requiresReview: bool = False  # camelCase for frontend

class WorkflowStepsInsert(BaseModel):
    steps: List[WorkflowStepCreate]

class DocumentCreate(BaseModel):
    name: str
    type: str
    size: Optional[int] = 0
    url: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    name: str
    type: str
    size: int
    url: Optional[str] = None
    confidence_score: Optional[float] = None
    uploaded_at: Optional[datetime]
    extracted_data: Optional[Dict[str, Any]] = {}

class AuditEntry(BaseModel):
    id: str
    action: str
    details: str
    performed_by: str
    step_number: Optional[int] = None
    confidence_score: Optional[float] = None
    sla_breached: bool = False
    timestamp: datetime

class LeaseRequestCreate(BaseModel):
    propertyId: str  # Match frontend field name
    tenantName: str  # Match frontend field name
    tenantABN: Optional[str] = ""  # Match frontend field name
    tenantACN: Optional[str] = ""  # Match frontend field name
    commencementDate: str  # Match frontend field name, accept string
    leaseTerm: int  # Match frontend field name
    rentAmount: float  # Match frontend field name
    securityDeposit: float  # Match frontend field name
    contactEmail: str  # Match frontend field name
    contactPhone: Optional[str] = ""  # Add missing field
    specialConditions: Optional[str] = ""  # Add missing field
    documents: Optional[List[DocumentCreate]] = []
    id: Optional[str] = None  # Frontend might send this, ignore it
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"

class LeaseRequestResponse(BaseModel):
    id: str
    propertyId: str  # camelCase for frontend
    propertyAddress: str  # camelCase for frontend
    tenantName: str  # camelCase for frontend
    tenantABN: Optional[str] = ""  # camelCase for frontend
    tenantACN: Optional[str] = ""  # camelCase for frontend
    requestorEmail: str  # camelCase for frontend
    commencementDate: date  # camelCase for frontend
    leaseTerm: int  # camelCase for frontend
    rentAmount: float  # camelCase for frontend
    securityDeposit: float  # camelCase for frontend
    status: str
    createdAt: datetime  # camelCase for frontend
    updatedAt: datetime  # camelCase for frontend
    documents: List[DocumentResponse] = []
    workflowSteps: List[WorkflowStepResponse] = []  # Updated to use proper model
    auditTrail: List[AuditEntry] = []  # camelCase for frontend

    class Config:
        from_attributes = True  # For SQLAlchemy compatibility

class PropertyResponse(BaseModel):
    id: str
    address: str
    unitNumber: Optional[str] = None  # camelCase for frontend
    propertyType: str  # camelCase for frontend
    usageType: Optional[str] = None  # camelCase for frontend
    availableArea: Optional[int] = None  # camelCase for frontend
    sapId: Optional[str] = None  # camelCase for frontend
    isAvailable: bool  # camelCase for frontend

class BusinessPartnerResponse(BaseModel):
    id: str
    name: str
    abn: Optional[str] = None
    acn: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    sapId: Optional[str] = None  # camelCase for frontend
    verified: bool
    createdAt: datetime  # camelCase for frontend

# ========================================
# FASTAPI APP SETUP
# ========================================

app = FastAPI(
    title="Lease Request Management API",
    description="Backend API for lease request processing and workflow management",
    version="1.0.0"
)

# CORS is now handled manually via middleware above
# Removed automatic CORS middleware to prevent conflicts

# ========================================
# MANUAL CORS HEADERS (Priority CORS handling)
# ========================================

@app.middleware("http")
async def add_cors_headers(request, call_next):
    # Handle preflight requests first
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response
    
    # Handle actual requests
    try:
        response = await call_next(request)
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        response = JSONResponse(
            content={"error": "Internal server error"}, 
            status_code=500
        )
    
    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

# Explicit OPTIONS handler for all routes
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    print(f"🔍 OPTIONS request for: /{full_path}")
    return JSONResponse(
        content={"message": "CORS preflight successful"}, 
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )


# ========================================
# DATABASE DEPENDENCY
# ========================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========================================
# UTILITY FUNCTIONS
# ========================================

def generate_lease_request_id() -> str:
    """Generate a unique lease request ID"""
    return f"LR{str(int(datetime.now().timestamp()))[4:]}"

def generate_document_id() -> str:
    """Generate a unique document ID"""
    return f"doc-{uuid.uuid4().hex[:8]}"

def generate_workflow_step_id() -> str:
    """Generate a unique workflow step ID"""
    return f"ws-{uuid.uuid4().hex[:8]}"

def generate_audit_id() -> str:
    """Generate a unique audit entry ID"""
    return f"audit-{uuid.uuid4().hex[:8]}"

def create_audit_entry(db: Session, lease_request_id: str, action: str, details: str, performed_by: str, step_number: int = None):
    """Create an audit entry for a lease request"""
    audit_entry = AuditEntryDB(
        id=generate_audit_id(),
        lease_request_id=lease_request_id,
        action=action,
        details=details,
        performed_by=performed_by,
        step_number=step_number,
        confidence_score=0.95
    )
    db.add(audit_entry)
    db.commit()
    return audit_entry


def create_workflow_steps(db: Session, lease_request_id: str) -> List[WorkflowStepDB]:
    """Create default workflow steps for a lease request"""
    
    # Define default workflow steps
    default_steps = [
        {
            "step_number": 1,
            "name": "Document Extraction",
            "description": "Extract and analyze uploaded documents",
            "status": "processing",
            "assigned_to": "system@example.com",
            "started_at": datetime.now()
        },
        {
            "step_number": 2,
            "name": "Validation Review", 
            "description": "Review extracted information for accuracy",
            "status": "pending",
            "assigned_to": "reviewer@example.com"
        },
        {
            "step_number": 3,
            "name": "Space Validation",
            "description": "Validate available space and property details",
            "status": "pending", 
            "assigned_to": "property@example.com"
        },
        {
            "step_number": 4,
            "name": "Business Partner Check",
            "description": "Verify tenant business partner information",
            "status": "pending",
            "assigned_to": "partner@example.com"
        },
        {
            "step_number": 5,
            "name": "ASIC Validation",
            "description": "Validate ABN and ACN through ASIC",
            "status": "pending",
            "assigned_to": "compliance@example.com"
        },
        {
            "step_number": 6,
            "name": "Shell Creation",
            "description": "Create lease shell in system",
            "status": "pending",
            "assigned_to": "system@example.com"
        },
        {
            "step_number": 7,
            "name": "Deposit Invoice",
            "description": "Generate and send deposit invoice",
            "status": "pending",
            "assigned_to": "finance@example.com"
        },
        {
            "step_number": 8,
            "name": "Abstract Verification",
            "description": "Verify lease abstract details",
            "status": "pending",
            "assigned_to": "legal@example.com"
        },
        {
            "step_number": 9,
            "name": "Clause Finalisation",
            "description": "Finalize lease clauses and terms",
            "status": "pending",
            "assigned_to": "legal@example.com"
        },
        {
            "step_number": 10,
            "name": "Activation",
            "description": "Activate the lease agreement",
            "status": "pending",
            "assigned_to": "manager@example.com"
        }
    ]
    
    created_steps = []
    
    try:
        for step_data in default_steps:
            workflow_step = WorkflowStepDB(
                id=generate_workflow_step_id(),
                lease_request_id=lease_request_id,
                step_number=step_data["step_number"],
                name=step_data["name"],
                description=step_data["description"],
                status=step_data["status"],
                assigned_to=step_data["assigned_to"],
                started_at=step_data.get("started_at"),
                confidence_score=0.95,
                requires_review=False
            )
            
            db.add(workflow_step)
            created_steps.append(workflow_step)
        
        db.commit()
        print(f"✅ Created {len(created_steps)} default workflow steps for lease request {lease_request_id}")
        return created_steps
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating workflow steps: {str(e)}")
        raise Exception(f"Failed to create workflow steps: {str(e)}")

# ========================================
# LEASE REQUESTS ENDPOINTS
# ========================================

@app.get("/api/lease-requests", response_model=List[LeaseRequestResponse])
async def get_all_lease_requests(db: Session = Depends(get_db)):
    """Fetch all lease requests"""
    try:
        print("🔍 GET /api/lease-requests - Starting request")
        lease_requests = db.query(LeaseRequestDB).all()
        print(f"🔍 Found {len(lease_requests)} lease requests in database")
        
        result = []
        for lr in lease_requests:
            print(f"🔍 Processing lease request: {lr.id}")
            
            # Convert documents
            documents = []
            for doc in lr.documents:
                documents.append(DocumentResponse(
                    id=doc.id,
                    name=doc.name,
                    type=doc.type,
                    size=doc.size or 0,
                    url=doc.url,
                    confidence_score=float(doc.confidence_score) if doc.confidence_score else 0.95,
                    uploaded_at=doc.uploaded_at,
                    extracted_data={}
                ))

            # Convert workflow steps safely
                workflow_steps = []
                try:
                    for ws in lr.workflow_steps:
                        workflow_step = WorkflowStepResponse(
                            id=ws.id,
                            stepNumber=ws.step_number,
                            name=ws.name,
                            description=ws.description,
                            status=ws.status,
                            startedAt=ws.started_at,
                            completedAt=ws.completed_at,
                            assignedTo=ws.assigned_to,
                            notes=ws.notes,
                            confidenceScore=float(ws.confidence_score) if ws.confidence_score else None,
                            requiresReview=ws.requires_review or False,
                            createdAt=ws.created_at,
                            updatedAt=ws.updated_at
                        )
                        workflow_steps.append(workflow_step)
                    print(f"✅ Processed {len(workflow_steps)} workflow steps")
                except Exception as ws_error:
                    print(f"❌ Error processing workflow steps: {str(ws_error)}")
                    workflow_steps = []  # Continue with empty workflow steps
            
            # Convert audit entries
            audit_entries = []
            for audit in lr.audit_entries:
                audit_entries.append(AuditEntry(
                    id=audit.id,
                    action=audit.action,
                    details=audit.details or "",
                    performed_by=audit.performed_by,
                    step_number=audit.step_number,
                    confidence_score=float(audit.confidence_score) if audit.confidence_score else 0.95,
                    sla_breached=audit.sla_breached or False,
                    timestamp=audit.timestamp
                ))
            
            result.append(LeaseRequestResponse(
                id=lr.id,
                propertyId=lr.property_id,  # Convert snake_case to camelCase
                propertyAddress=lr.property_address,
                tenantName=lr.tenant_name,
                tenantABN=lr.tenant_abn or "",
                tenantACN=lr.tenant_acn or "",
                requestorEmail=lr.requestor_email,
                commencementDate=lr.commencement_date,
                leaseTerm=lr.lease_term,
                rentAmount=float(lr.rent_amount),
                securityDeposit=float(lr.security_deposit),
                status=lr.status,
                createdAt=lr.created_at,
                updatedAt=lr.updated_at,
                documents=documents,
                workflowSteps=workflow_steps,
                auditTrail=audit_entries
            ))
        
        print(f"✅ Successfully processed {len(result)} lease requests")
        return result
        
    except Exception as e:
        print(f"❌ Error in get_all_lease_requests: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/lease-requests/{request_id}", response_model=LeaseRequestResponse)
async def get_lease_request_by_id(request_id: str, db: Session = Depends(get_db)):
    """Fetch a single lease request by ID"""
    lr = db.query(LeaseRequestDB).filter(LeaseRequestDB.id == request_id).first()
    
    if not lr:
        raise HTTPException(status_code=404, detail="Lease request not found")
    
    # Convert documents
    documents = []
    for doc in lr.documents:
        documents.append(DocumentResponse(
            id=doc.id,
            name=doc.name,
            type=doc.type,
            size=doc.size or 0,
            url=doc.url,
            confidence_score=float(doc.confidence_score) if doc.confidence_score else 0.95,
            uploaded_at=doc.uploaded_at,
            extracted_data={}
        ))

    # Convert workflow steps safely
    workflow_steps = []
    try:
        for ws in lr.workflow_steps:
            workflow_step = WorkflowStepResponse(
                id=ws.id,
                stepNumber=ws.step_number,
                name=ws.name,
                description=ws.description,
                status=ws.status,
                startedAt=ws.started_at,
                completedAt=ws.completed_at,
                assignedTo=ws.assigned_to,
                notes=ws.notes,
                confidenceScore=float(ws.confidence_score) if ws.confidence_score else None,
                requiresReview=ws.requires_review or False,
                createdAt=ws.created_at,
                updatedAt=ws.updated_at
                )
        workflow_steps.append(workflow_step)
        print(f"✅ Processed {len(workflow_steps)} workflow steps")
    except Exception as ws_error:
        print(f"❌ Error processing workflow steps: {str(ws_error)}")
        workflow_steps = []  # Continue with empty workflow steps
            
    
    # Convert audit entries
    audit_entries = []
    for audit in lr.audit_entries:
        audit_entries.append(AuditEntry(
            id=audit.id,
            action=audit.action,
            details=audit.details or "",
            performed_by=audit.performed_by,
            step_number=audit.step_number,
            confidence_score=float(audit.confidence_score) if audit.confidence_score else 0.95,
            sla_breached=audit.sla_breached or False,
            timestamp=audit.timestamp
        ))
    
    return LeaseRequestResponse(
        id=lr.id,
        propertyId=lr.property_id,
        propertyAddress=lr.property_address,
        tenantName=lr.tenant_name,
        tenantABN=lr.tenant_abn or "",
        tenantACN=lr.tenant_acn or "",
        requestorEmail=lr.requestor_email,
        commencementDate=lr.commencement_date,
        leaseTerm=lr.lease_term,
        rentAmount=float(lr.rent_amount),
        securityDeposit=float(lr.security_deposit),
        status=lr.status,
        createdAt=lr.created_at,
        updatedAt=lr.updated_at,
        documents=documents,
        workflowSteps=workflow_steps,
        auditTrail=audit_entries
    )

@app.post("/api/lease-requests", response_model=LeaseRequestResponse)
async def create_lease_request(request_data: LeaseRequestCreate, db: Session = Depends(get_db)):
    """Create a new lease request"""
    
    try:
        # Convert field names to match database
        property_id = request_data.propertyId
        print(f"🔍 Looking for property with ID: {property_id}")
        
        # Get property details
        property_obj = db.query(PropertyDB).filter(PropertyDB.id == property_id).first()
        if not property_obj:
            # List available properties for debugging
            available_properties = db.query(PropertyDB).all()
            available_ids = [p.id for p in available_properties]
            print(f"❌ Property {property_id} not found. Available properties: {available_ids}")
            raise HTTPException(
                status_code=400, 
                detail=f"Property '{property_id}' not found. Available properties: {available_ids}"
            )
        
        print(f"✅ Found property: {property_obj.address}")
        
        tenant_name = request_data.tenantName
        tenant_abn = request_data.tenantABN
        tenant_acn = request_data.tenantACN
        contact_email = request_data.contactEmail
        lease_term = request_data.leaseTerm
        rent_amount = request_data.rentAmount
        security_deposit = request_data.securityDeposit
        
        # Parse the date string (handle both ISO format and simple date format)
        try:
            date_str = request_data.commencementDate
            # Handle ISO format like "2025-09-22T16:00:00.000Z"
            if "T" in date_str:
                commencement_date = datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
            else:
                # Handle simple format like "2025-09-22"
                commencement_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            print(f"✅ Parsed date: {commencement_date}")
        except ValueError as e:
            print(f"❌ Date parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}. Expected ISO format or YYYY-MM-DD")
        
        
        # Generate unique ID
        request_id = generate_lease_request_id()
        print(f"🔍 Creating lease request with ID: {request_id}")
        
        # Create lease request
        lease_request = LeaseRequestDB(
            id=request_id,
            property_id=property_id,
            tenant_name=tenant_name,
            tenant_abn=tenant_abn,
            tenant_acn=tenant_acn,
            property_address=property_obj.address,
            lease_term=lease_term,
            commencement_date=commencement_date,
            rent_amount=rent_amount,
            security_deposit=security_deposit,
            status="document_extraction",
            requestor_email=contact_email
        )
        
        db.add(lease_request)
        db.commit()
        db.refresh(lease_request)
        
        print("request_data.documents")
        print(request_data.documents)

        # Create documents
        documents = []
        for i, doc_data in enumerate(request_data.documents):
            document = LeaseDocumentDB(
                id=generate_document_id(),
                lease_request_id=request_id,
                name=doc_data.name,
                type=doc_data.type,
                size=doc_data.size,
                url=f"#document-{i}",
                confidence_score=0.95
            )
            db.add(document)
            documents.append(DocumentResponse(
                id=document.id,
                name=document.name,
                type=document.type,
                size=document.size or 0,
                url=document.url,
                confidence_score=0.95,
                uploaded_at=document.uploaded_at,
                extracted_data={}
            ))

        db.commit()
        
        # Create default workflow steps
        try:
            workflow_steps_db = create_workflow_steps(db, request_id)
            print(f"✅ Created {len(workflow_steps_db)} workflow steps")
            
            # Convert to response format
            workflow_steps = []
            for ws in workflow_steps_db:
                workflow_steps.append(WorkflowStepResponse(
                    id=ws.id,
                    stepNumber=ws.step_number,
                    name=ws.name,
                    description=ws.description,
                    status=ws.status,
                    startedAt=ws.started_at,
                    completedAt=ws.completed_at,
                    assignedTo=ws.assigned_to,
                    notes=ws.notes,
                    confidenceScore=float(ws.confidence_score) if ws.confidence_score else None,
                    requiresReview=ws.requires_review or False,
                    createdAt=ws.created_at,
                    updatedAt=ws.updated_at
                ))
        except Exception as workflow_error:
            print(f"❌ Warning: Failed to create workflow steps: {str(workflow_error)}")
            workflow_steps = []  # Continue without workflow steps if creation fails
        
        # Create audit entry
        audit_entry = create_audit_entry(
            db=db,
            lease_request_id=request_id,
            action="Request Submitted",
            details=f"Lease request submitted for {tenant_name}",
            performed_by=contact_email,
            step_number=1
        )
        
        audit_entries = [AuditEntry(
            id=audit_entry.id,
            action=audit_entry.action,
            details=audit_entry.details or "",
            performed_by=audit_entry.performed_by,
            step_number=audit_entry.step_number,
            confidence_score=float(audit_entry.confidence_score) if audit_entry.confidence_score else 0.95,
            sla_breached=audit_entry.sla_breached or False,
            timestamp=audit_entry.timestamp
        )]
        
        return LeaseRequestResponse(
            id=lease_request.id,
            propertyId=lease_request.property_id,
            propertyAddress=lease_request.property_address,
            tenantName=lease_request.tenant_name,
            tenantABN=lease_request.tenant_abn or "",
            tenantACN=lease_request.tenant_acn or "",
            requestorEmail=lease_request.requestor_email,
            commencementDate=lease_request.commencement_date,
            leaseTerm=lease_request.lease_term,
            rentAmount=float(lease_request.rent_amount),
            securityDeposit=float(lease_request.security_deposit),
            status=lease_request.status,
            createdAt=lease_request.created_at,
            updatedAt=lease_request.updated_at,
            documents=documents,
            workflowSteps=workflow_steps,  # Now includes default workflow steps
            auditTrail=audit_entries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating lease request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.patch("/api/lease-requests/{request_id}", response_model=LeaseRequestResponse)
async def update_lease_request(request_id: str, update_data: dict, db: Session = Depends(get_db)):
    """Update a lease request"""
    lease_request = db.query(LeaseRequestDB).filter(LeaseRequestDB.id == request_id).first()
    
    if not lease_request:
        raise HTTPException(status_code=404, detail="Lease request not found")
    
    # Update fields
    for field, value in update_data.items():
        if hasattr(lease_request, field):
            setattr(lease_request, field, value)
    
    lease_request.updated_at = datetime.now()
    db.commit()
    db.refresh(lease_request)
    
    # Create audit entry for update
    create_audit_entry(
        db=db,
        lease_request_id=request_id,
        action="Request Updated",
        details=f"Lease request updated: {', '.join(update_data.keys())}",
        performed_by="system@example.com"  # You might want to get this from auth context
    )
    
    # Return updated request (you can reuse the get endpoint logic here)
    return await get_lease_request_by_id(request_id, db)


# ========================================
# WORKFLOW STEPS ENDPOINTS
# ========================================

@app.get("/api/lease-requests/{request_id}/workflow-steps", response_model=List[WorkflowStepResponse])
async def get_workflow_steps(request_id: str, db: Session = Depends(get_db)):
    """Fetch all workflow steps for a lease request"""
    try:
        print(f"🔍 GET /api/lease-requests/{request_id}/workflow-steps")
        
        # Check if lease request exists
        lease_request = db.query(LeaseRequestDB).filter(LeaseRequestDB.id == request_id).first()
        if not lease_request:
            raise HTTPException(status_code=404, detail="Lease request not found")
        
        # Get workflow steps
        workflow_steps = db.query(WorkflowStepDB).filter(
            WorkflowStepDB.lease_request_id == request_id
        ).order_by(WorkflowStepDB.step_number).all()
        
        result = []
        for ws in workflow_steps:
            result.append(WorkflowStepResponse(
                id=ws.id,
                stepNumber=ws.step_number,
                name=ws.name,
                description=ws.description,
                status=ws.status,
                startedAt=ws.started_at,
                completedAt=ws.completed_at,
                assignedTo=ws.assigned_to,
                notes=ws.notes,
                confidenceScore=float(ws.confidence_score) if ws.confidence_score else None,
                requiresReview=ws.requires_review or False,
                createdAt=ws.created_at,
                updatedAt=ws.updated_at
            ))
        
        print(f"✅ Found {len(result)} workflow steps for lease request {request_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching workflow steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

#Not being used
@app.post("/api/lease-requests/{request_id}/workflow-steps")
async def insert_workflow_steps(request_id: str, workflow_data: WorkflowStepsInsert, db: Session = Depends(get_db)):
    """Insert workflow steps for a lease request"""
    try:

        print(f"🔍 POST /api/lease-requests/{request_id}/workflow-steps")
        print(f"🔍 Inserting {len(workflow_data.steps)} workflow steps")

        # Check if lease request exists
        lease_request = db.query(LeaseRequestDB).filter(LeaseRequestDB.id == request_id).first()
        if not lease_request:
            raise HTTPException(status_code=404, detail="Lease request not found")
        
        # Check if workflow steps already exist for this lease request
        existing_steps = db.query(WorkflowStepDB).filter(
            WorkflowStepDB.lease_request_id == request_id
        ).count()
        
        if existing_steps > 0:
            raise HTTPException(status_code=400, detail="Workflow steps already exist for this lease request")
        
        # Insert workflow steps
        created_steps = []
        for step_data in workflow_data.steps:
            workflow_step = WorkflowStepDB(
                id=generate_workflow_step_id(),
                lease_request_id=request_id,
                step_number=step_data.stepNumber,
                name=step_data.name,
                description=step_data.description,
                status=step_data.status,
                assigned_to=step_data.assignedTo,
                notes=step_data.notes,
                confidence_score=step_data.confidenceScore,
                requires_review=step_data.requiresReview,
                started_at=datetime.now() if step_data.status == 'in_progress' else None
            )
            
            db.add(workflow_step)
            created_steps.append(workflow_step)
        
        db.commit()
        
        print(f"✅ Successfully inserted {len(created_steps)} workflow steps")
        return {"success": True, "message": f"Inserted {len(created_steps)} workflow steps"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error inserting workflow steps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.patch("/api/lease-requests/{request_id}/workflow-steps/{step_id}")
async def update_workflow_step(request_id: str, step_id: str, update_data: dict, db: Session = Depends(get_db)):
    """Update a specific workflow step"""
    try:
        print(f"🔍 PATCH /api/lease-requests/{request_id}/workflow-steps/{step_id}")
        
        # Find the workflow step
        workflow_step = db.query(WorkflowStepDB).filter(
            WorkflowStepDB.step_number == step_id,
            WorkflowStepDB.lease_request_id == request_id
        ).first()
        
        if not workflow_step:
            raise HTTPException(status_code=404, detail="Workflow step not found")
        
        # Update fields
        for field, value in update_data.items():
            # Convert camelCase to snake_case for database fields
            db_field = field
            if field == 'stepNumber':
                db_field = 'step_number'
            elif field == 'startedAt':
                db_field = 'started_at'
                if value:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            elif field == 'completedAt':
                db_field = 'completed_at'
                if value:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            elif field == 'assignedTo':
                db_field = 'assigned_to'
            elif field == 'confidenceScore':
                db_field = 'confidence_score'
            elif field == 'requiresReview':
                db_field = 'requires_review'
            
            if hasattr(workflow_step, db_field):
                setattr(workflow_step, db_field, value)
        
        workflow_step.updated_at = datetime.now()
        db.commit()
        
        # Return updated workflow step
        return WorkflowStepResponse(
            id=workflow_step.id,
            stepNumber=workflow_step.step_number,
            name=workflow_step.name,
            description=workflow_step.description,
            status=workflow_step.status,
            startedAt=workflow_step.started_at,
            completedAt=workflow_step.completed_at,
            assignedTo=workflow_step.assigned_to,
            notes=workflow_step.notes,
            confidenceScore=float(workflow_step.confidence_score) if workflow_step.confidence_score else None,
            requiresReview=workflow_step.requires_review or False,
            createdAt=workflow_step.created_at,
            updatedAt=workflow_step.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error updating workflow step: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ========================================
# PROPERTIES ENDPOINTS
# ========================================

@app.get("/api/properties", response_model=List[PropertyResponse])
async def get_all_properties(db: Session = Depends(get_db)):
    """Fetch all properties"""
    properties = db.query(PropertyDB).all()
    
    return [PropertyResponse(
        id=prop.id,
        address=prop.address,
        unitNumber=prop.unit_number,
        propertyType=prop.property_type,
        usageType=prop.usage_type,
        availableArea=prop.available_area,
        sapId=prop.sap_id,
        isAvailable=prop.is_available
    ) for prop in properties]

@app.get("/api/properties/{property_id}", response_model=PropertyResponse)
async def get_property_by_id(property_id: str, db: Session = Depends(get_db)):
    """Fetch a single property by ID"""
    property_obj = db.query(PropertyDB).filter(PropertyDB.id == property_id).first()
    
    if not property_obj:
        raise HTTPException(status_code=404, detail="Property not found")
    
    return PropertyResponse(
        id=property_obj.id,
        address=property_obj.address,
        unitNumber=property_obj.unit_number,
        propertyType=property_obj.property_type,
        usageType=property_obj.usage_type,
        availableArea=property_obj.available_area,
        sapId=property_obj.sap_id,
        isAvailable=property_obj.is_available
    )

# ========================================
# BUSINESS PARTNERS ENDPOINTS - NOT BEING USED
# ========================================

@app.get("/api/business-partners", response_model=List[BusinessPartnerResponse])
async def get_all_business_partners(db: Session = Depends(get_db)):
    """Fetch all business partners"""
    partners = db.query(BusinessPartnerDB).all()
    
    return [BusinessPartnerResponse(
        id=partner.id,
        name=partner.name,
        abn=partner.abn,
        acn=partner.acn,
        address=partner.address,
        phone=partner.phone,
        email=partner.email,
        sapId=partner.sap_id,
        verified=partner.verified,
        createdAt=partner.created_at
    ) for partner in partners]

# ========================================
# HEALTH CHECK
# ========================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Lease Request Management API", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3002)
