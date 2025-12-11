# Open-NueChat User Guide

A comprehensive guide to using Open-NueChat's features including Custom GPTs, Knowledge Bases, the Marketplace, and more.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Chat Basics](#chat-basics)
- [Model Selection](#model-selection)
- [Custom GPTs (Assistants)](#custom-gpts-assistants)
- [Knowledge Bases](#knowledge-bases)
- [Marketplace](#marketplace)
- [Voice Features](#voice-features)
- [Image Generation](#image-generation)
- [Tools](#tools)
- [Account & Billing](#account--billing)

---

## Getting Started

### Creating an Account

1. Navigate to the Open-NueChat URL (default: `http://localhost:8000`)
2. Click **Sign Up** or **Register**
3. Enter your email, username, and password
4. Alternatively, use **Sign in with Google** or **Sign in with GitHub** if OAuth is enabled

### Logging In

1. Click **Sign In**
2. Enter your email and password
3. Or use OAuth providers (Google/GitHub)

### First Chat

1. After logging in, you'll see the chat interface
2. Type a message in the input box at the bottom
3. Press **Enter** or click **Send**
4. The AI will respond in real-time with streaming text

---

## Chat Basics

### Starting a New Chat

- Click the **New Chat** button (+ icon) in the sidebar
- Or use keyboard shortcut: `Ctrl/Cmd + N`

### Chat Features

| Feature | How to Use |
|---------|------------|
| **Send Message** | Type and press Enter |
| **New Line** | Shift + Enter |
| **Regenerate** | Click the regenerate icon on any assistant message |
| **Edit Message** | Click the edit icon on your message, modify, and resend |
| **Branch Conversation** | Regenerate creates a new branch; swipe left/right to view alternatives |
| **Copy Message** | Click the copy icon on any message |
| **Delete Chat** | Click the trash icon in the sidebar or chat header |

### Uploading Files

1. Click the **attachment icon** (📎) in the chat input
2. Select files to upload:
   - **Documents**: PDF, TXT, MD, DOCX, CSV, JSON
   - **Images**: PNG, JPG, GIF (for vision-capable models)
   - **Code**: ZIP files with full project context
3. Files are processed and included in the conversation context

### Exporting & Sharing

1. Open the chat you want to share
2. Click the **Share** icon in the chat header
3. Options:
   - **Copy Link**: Creates a public shareable link
   - **Export JSON**: Downloads the full conversation
   - **Export Markdown**: Downloads as formatted text

---

## Model Selection

### Selecting a Model

1. Click the **model selector** dropdown at the top of the chat
2. Choose from available models:
   - **Base Models**: Direct LLM models (e.g., Llama, Mistral, GPT)
   - **Custom GPTs**: Your created assistants
   - **Subscribed GPTs**: Assistants you've subscribed to from the marketplace

### Model Categories

| Category | Description |
|----------|-------------|
| **Models** | Raw LLM models from your backend (Ollama, vLLM, etc.) |
| **My GPTs** | Custom assistants you've created |
| **Subscribed** | Marketplace GPTs you've subscribed to |

### Setting a Default Model

1. Go to **Settings** → **Preferences**
2. Under **Default Model**, select your preferred model
3. New chats will start with this model

---

## Custom GPTs (Assistants)

Custom GPTs are personalized AI assistants with specific instructions, personalities, and optional knowledge bases.

### Creating a Custom GPT

1. Click **GPTs** in the sidebar (or navigate to `/assistants`)
2. Click **Create New GPT** (+ button)
3. Fill in the configuration:

#### Basic Settings

| Field | Description | Example |
|-------|-------------|---------|
| **Name** | Display name for your GPT | "Python Tutor" |
| **Description** | Brief description of what it does | "Helps learn Python programming with examples" |
| **Avatar** | Upload an image or use emoji | 🐍 |

#### Personality & Instructions

| Field | Description |
|-------|-------------|
| **System Prompt** | Core instructions that define behavior |
| **Personality** | Tone and style (friendly, professional, casual) |
| **Welcome Message** | First message shown when starting a chat |
| **Suggested Prompts** | Quick-start questions shown to users |

**Example System Prompt:**
```
You are a friendly Python programming tutor. You:
- Explain concepts clearly with simple examples
- Use analogies to make complex topics accessible
- Always provide working code examples
- Encourage questions and celebrate progress
- Point out common mistakes and best practices
```

#### Model Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Base Model** | Which LLM to use | (system default) |
| **Temperature** | Creativity (0.0-2.0) | 0.7 |
| **Max Tokens** | Maximum response length | 4096 |
| **Top P** | Nucleus sampling | 1.0 |

#### Knowledge Base

1. Under **Knowledge**, click **Add Knowledge Base**
2. Select existing knowledge bases or create new ones
3. The GPT will search these documents when responding

### Editing a Custom GPT

1. Go to **GPTs** in the sidebar
2. Find your GPT and click **Edit** (pencil icon)
3. Modify settings and click **Save**

### Deleting a Custom GPT

1. Go to **GPTs** in the sidebar
2. Find your GPT and click **Delete** (trash icon)
3. Confirm deletion

> **Note**: Deleting a published GPT removes it from the marketplace

### Using Your Custom GPT

1. In any chat, click the **model selector**
2. Under **My GPTs**, select your assistant
3. Start chatting - it will use your configured personality and knowledge

---

## Knowledge Bases

Knowledge Bases enable RAG (Retrieval-Augmented Generation) - your GPT can search through your documents to provide accurate, contextual answers.

### Creating a Knowledge Base

1. Click **Knowledge** in the sidebar (or navigate to `/knowledge`)
2. Click **Create Knowledge Base** (+ button)
3. Configure:

| Field | Description |
|-------|-------------|
| **Name** | Descriptive name | "Company Policies" |
| **Description** | What documents it contains | "HR policies, employee handbook, benefits info" |
| **Visibility** | Private (only you) or Public (marketplace) | Private |

4. Click **Create**

### Uploading Documents

1. Open your knowledge base
2. Click **Upload Documents** or drag-and-drop files
3. Supported formats:

| Format | Extensions |
|--------|------------|
| **Documents** | PDF, DOCX, DOC, TXT, MD, RTF |
| **Data** | CSV, JSON, JSONL, XML |
| **Code** | PY, JS, TS, HTML, CSS, and 40+ languages |
| **Archives** | ZIP (extracts and processes all files) |

4. Documents are queued for processing:
   - Text extraction
   - Chunking (splitting into searchable segments)
   - Embedding generation
   - Vector indexing

### Document Processing Status

| Status | Meaning |
|--------|---------|
| **Queued** | Waiting to be processed |
| **Processing** | Currently being embedded |
| **Ready** | Available for search |
| **Error** | Processing failed (click for details) |

### Managing Documents

- **View**: Click a document to see its chunks
- **Delete**: Remove a document and its embeddings
- **Reprocess**: Re-extract and re-embed a document

### Attaching Knowledge Bases to GPTs

1. Edit your Custom GPT
2. Scroll to **Knowledge Bases**
3. Click **Add** and select knowledge bases
4. Save the GPT

When chatting, the GPT will automatically search relevant documents.

### Knowledge Base Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Chunk Size** | Characters per chunk | 512 |
| **Chunk Overlap** | Overlap between chunks | 50 |
| **Top K Results** | Documents returned per search | 5 |

---

## Marketplace

The Marketplace allows you to discover and share Custom GPTs and Knowledge Bases.

### Browsing the Marketplace

1. Click **Explore** in the sidebar (or navigate to `/explore`)
2. Browse categories:
   - **Featured**: Staff picks and popular GPTs
   - **New**: Recently published
   - **Categories**: Writing, Coding, Research, etc.
3. Use the **search bar** to find specific GPTs

### Subscribing to a GPT

1. Find a GPT you want to use
2. Click on it to view details:
   - Description and capabilities
   - User ratings and reviews
   - Creator information
3. Click **Subscribe**
4. The GPT appears in your model selector under **Subscribed**

### Unsubscribing

1. Go to **GPTs** → **Subscribed** tab
2. Find the GPT and click **Unsubscribe**
3. Or: In the marketplace, open the GPT and click **Unsubscribe**

### Rating & Reviewing

1. After using a subscribed GPT, go to its marketplace page
2. Click **Rate & Review**
3. Select 1-5 stars
4. Optionally write a review
5. Submit

---

## Publishing to the Marketplace

Share your Custom GPTs and Knowledge Bases with the community.

### Publishing a Custom GPT

1. Go to **GPTs** in the sidebar
2. Find your GPT and click **Edit**
3. Scroll to **Publishing**
4. Configure:

| Field | Description |
|-------|-------------|
| **Visibility** | Public (marketplace) or Private |
| **Category** | Select the best fit (Writing, Coding, etc.) |
| **Tags** | Keywords for discoverability |
| **Pricing** | Free or token cost per use (if billing enabled) |

5. Click **Publish**

### Publishing Requirements

Before publishing, ensure your GPT has:
- ✅ Clear, descriptive name
- ✅ Helpful description
- ✅ Avatar image or emoji
- ✅ Welcome message
- ✅ At least 2-3 suggested prompts
- ✅ Well-crafted system prompt

### Updating a Published GPT

1. Edit your GPT normally
2. Changes are reflected immediately in the marketplace
3. Existing subscribers see the updated version

### Unpublishing

1. Edit your GPT
2. Set **Visibility** to **Private**
3. Save - the GPT is removed from the marketplace
4. Existing subscribers lose access

### Publishing a Knowledge Base

1. Go to **Knowledge** in the sidebar
2. Find your knowledge base and click **Edit**
3. Set **Visibility** to **Public**
4. Add description and tags
5. Save

Public knowledge bases can be:
- Discovered in the marketplace
- Attached to other users' GPTs (if permitted)

---

## Voice Features

### Text-to-Speech (TTS)

Listen to AI responses:

1. Hover over any assistant message
2. Click the **speaker icon** (🔊)
3. Audio plays automatically
4. Controls: Play/Pause, Speed adjustment

### Speech-to-Text (STT)

Speak your messages:

1. Click the **microphone icon** (🎤) in the chat input
2. Allow microphone access if prompted
3. Speak your message
4. Click again to stop recording
5. Your speech is transcribed and sent

### Talk-to-Me Mode

Hands-free conversation:

1. Click **Talk to Me** button (or use keyboard shortcut)
2. The AI speaks its responses automatically
3. Voice Activity Detection (VAD) listens for your speech
4. Speak naturally - it detects when you start and stop
5. Click again to exit Talk-to-Me mode

---

## Image Generation

Generate images directly in chat (requires image service).

### Generating Images

1. In chat, ask the AI to create an image:
   - "Generate an image of a sunset over mountains"
   - "Create a logo for a coffee shop called 'Bean There'"
   - "Draw a cartoon cat wearing a hat"

2. The AI uses the image generation tool
3. Image appears in the chat when ready
4. Click to view full size or download

### Image Options

You can specify:
- **Style**: "photorealistic", "cartoon", "watercolor", "pixel art"
- **Aspect Ratio**: "square", "landscape", "portrait"
- **Details**: Colors, mood, specific elements

**Example prompts:**
```
Create a photorealistic image of a cozy cabin in the woods during autumn
Generate a minimalist logo with the letter 'A' in blue and white
Draw a cute cartoon robot holding a flower, in a friendly style
```

---

## Tools

Custom GPTs and the base chat can use built-in tools.

### Available Tools

| Tool | Description | Example Use |
|------|-------------|-------------|
| **calculator** | Math calculations | "What's 15% of 847?" |
| **get_current_time** | Current date/time | "What time is it in Tokyo?" |
| **search_documents** | Search knowledge bases | (automatic with RAG) |
| **execute_python** | Run Python code | "Calculate the first 20 Fibonacci numbers" |
| **format_json** | Format/validate JSON | "Format this JSON: {messy json}" |
| **analyze_text** | Text statistics | "Analyze the readability of this paragraph" |
| **web_search** | Search the web | "What's the latest news about AI?" |

### Tool Usage

Tools are used automatically when relevant. The AI decides when to use them based on your request.

**Examples:**
- "What's 2^10?" → Uses calculator
- "Search my documents for the refund policy" → Uses search_documents
- "Write a Python script to sort a list" → May use execute_python to verify

---

## Account & Billing

### Profile Settings

1. Click your **avatar** in the top-right
2. Select **Settings**
3. Edit:
   - Display name
   - Email
   - Password
   - Avatar

### Token Usage

Track your AI usage:

1. Go to **Settings** → **Usage**
2. View:
   - Tokens used this period
   - Token limit for your tier
   - Reset date

### Subscription Tiers

| Tier | Token Limit | Features |
|------|-------------|----------|
| **Free** | 100,000/month | Basic chat, 1 GPT, 1 Knowledge Base |
| **Pro** | 1,000,000/month | Unlimited GPTs, 10 Knowledge Bases, Priority |
| **Enterprise** | 10,000,000/month | Everything + API access, Team features |

### Upgrading

1. Go to **Settings** → **Subscription**
2. Select a plan
3. Enter payment details (if Stripe billing enabled)
4. Confirm upgrade

### API Keys

For developers:

1. Go to **Settings** → **API Keys**
2. Click **Generate New Key**
3. Set scopes (read, write, admin)
4. Copy the key (shown only once)
5. Use in API requests: `Authorization: Bearer <key>`

---

## Tips & Best Practices

### Creating Effective GPTs

1. **Be specific in system prompts** - Clear instructions produce better results
2. **Use examples** - Show the GPT how to respond
3. **Set constraints** - Define what it should NOT do
4. **Test thoroughly** - Try edge cases before publishing
5. **Iterate** - Refine based on actual usage

### Building Quality Knowledge Bases

1. **Organize documents** - Group related content
2. **Clean data** - Remove irrelevant content before uploading
3. **Use descriptive filenames** - Helps with search relevance
4. **Update regularly** - Keep information current
5. **Test searches** - Verify relevant results are returned

### Getting Better Responses

1. **Be specific** - "Write a 200-word summary" vs "Summarize this"
2. **Provide context** - Background info helps accuracy
3. **Use follow-ups** - Refine with additional instructions
4. **Try regenerate** - Different responses may be better
5. **Switch models** - Some models excel at certain tasks

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + N` | New chat |
| `Ctrl/Cmd + K` | Search chats |
| `Ctrl/Cmd + /` | Toggle sidebar |
| `Enter` | Send message |
| `Shift + Enter` | New line in message |
| `Escape` | Cancel current action |
| `↑` | Edit last message |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Chat not responding** | Check if LLM backend is running |
| **Document stuck processing** | Check document queue in admin panel |
| **Knowledge search no results** | Verify documents are in "Ready" status |
| **TTS not working** | Ensure TTS service is started |
| **Image generation fails** | Check image service status |

### Getting Help

- Check the **Admin Panel** for system status
- View **logs**: `./control.sh logs -f`
- Report issues on GitHub

---

## Glossary

| Term | Definition |
|------|------------|
| **GPT** | Custom AI assistant with specific instructions |
| **RAG** | Retrieval-Augmented Generation - searching documents to enhance responses |
| **Knowledge Base** | Collection of documents for RAG |
| **Embedding** | Vector representation of text for semantic search |
| **Token** | Unit of text (roughly 4 characters or 0.75 words) |
| **System Prompt** | Instructions that define AI behavior |
| **Temperature** | Controls randomness (0=deterministic, 2=creative) |
