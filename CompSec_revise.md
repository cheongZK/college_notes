Okay, this is a comprehensive request! Let's break down the BAIT1093 Introduction to Computer Security syllabus into revision notes, flashcards, mini-tests, and mock exam questions, focusing on:

1.  **Access Control** (primarily from Chapter 4, with support from Chapter 6 for physical aspects)
2.  **About Malware** (primarily from Chapter 2)
3.  **Basics of Cryptography** (primarily from Chapter 3)

I'll use the provided OCR text from your slides as the primary source.

---

## Part 1: Revision Notes

---

### **Topic 1: Access Control**

**I. Introduction to Access Control (Chapter 4, Slide 82-83, Chapter 1 Slide 10-12)**

*   **Definition (NIST IR 7298):** The process of granting or denying specific requests to:
    *   Obtain and use information and related processing services.
    *   Enter specific physical facilities.
*   **Definition (RFC 4949):** A process by which use of system resources is regulated according to a security policy and is permitted only by authorized entities (users, programs, processes, or other systems) according to that policy.
*   **Purpose:** Implements a security policy specifying who or what can access which system resources (applications, OS, firewalls, files, databases) and the type of access permitted.
*   **Core Components (Chapter 4, Slide 84):**
    *   **Authentication:** Verifying the credentials of a user or system entity are valid. *This is the prerequisite for access control.*
    *   **Authorization:** Granting a right or permission to a system entity to access a system resource. Determines who is trusted for a given purpose.
    *   **Audit:** Independent review and examination of system records and activities to test controls, ensure policy compliance, detect breaches, and recommend improvements.
*   **Process Flow (Chapter 4, Slide 85):**
    1.  System authenticates an entity seeking access.
    2.  Access control function determines if the specific requested access is permitted.
    3.  A security administrator maintains an authorization database.
    4.  Access control function consults this database.
    5.  Auditing function monitors and records user access.
*   **Key Terminology (Chapter 4, Slide 88):**
    *   **Subject:** Entity capable of accessing objects (Owner, Group, World).
    *   **Object:** Resource to which access is controlled (e.g., file, program).
    *   **Access Right:** The way a subject may access an object (Read, Write, Execute, Delete, Create, Search).

**II. Authentication (Chapter 4)**

*   **Definition:** The process of verifying an identity claimed by or for a system entity (RFC 4949).
*   **Fundamental building block and primary line of defense.**
*   **Basis for most access control and user accountability.**
*   **Two-Step Process (Chapter 4, Slide 4):**
    1.  **Identification Step:** Presenting an identifier (e.g., User ID).
    2.  **Verification Step:** Presenting/generating authentication information corroborating the identifier (e.g., Password).
*   **Means of Authentication (Chapter 4, Slide 5):**
    1.  **Something the individual knows:** Passwords, PINs, security questions.
    2.  **Something the individual possesses (token):** Smartcards, electronic keycards, physical keys, security tokens.
    3.  **Something the individual is (static biometrics):** Fingerprints, retina, face.
    4.  **Something the individual does (dynamic/behavioral biometrics):** Voice pattern, handwriting, typing rhythm.

    *   **A. Password Authentication (Chapter 4, Slides 6-29):**
        *   Most common IT authentication credential.
        *   **Challenges:** Memorization, unique passwords for multiple accounts, password expiration policies.
        *   **User Shortcuts:** Weak passwords (common words, short, predictable, personal info), password reuse.
        *   **Predictable Patterns:** Appending numbers/symbols, replacing letters (e.g., 'o' with '0').
        *   **Attacks on Passwords:**
            *   **Pass the Hash Attack/Password Cracker:** Stealing password digests (hashes).
                *   *Pass the Hash:* Uses stolen hash to impersonate (e.g., NTLM hash).
                *   *Password Cracker:* Creates candidate digests and compares against stolen ones.
            *   **Password Spraying:** Trying a few common passwords against many user accounts.
            *   **Brute Force Attack:** Trying every possible combination. Online (pounds one account) or Offline (uses stolen digest file).
            *   **Rule Attack:** Statistical analysis on stolen passwords to create masks (?u?l?l?l?d?d?d?d) to guide cracking.
            *   **Dictionary Attack:** Creating digests of common dictionary words.
            *   **Rainbow Tables:** Large pre-generated dataset of candidate digests (chains), faster than dictionary attacks, less memory.
            *   **Password Collections:** Using publicly leaked passwords as candidates.
        *   **Protecting Password Digests (Password Security Solutions - Chapter 4, Slides 75-76):**
            *   **Salts:** Random strings added to plaintext passwords *before* hashing. Makes each digest unique even for same passwords, prevents pre-computed table attacks (like rainbow tables for unsalted hashes).
            *   **Key Stretching:** Slows down password hashing (e.g., bcrypt, PBKDF2, Argon2) by increasing computation time, making brute-force attacks harder.
        *   **Managing Passwords (Chapter 4, Slides 77-79):**
            *   **Length over complexity:** Longer passwords are harder to crack.
            *   **Password Vaults:** Secure repositories (generators, online vaults, management apps).
            *   **Password Keys:** Hardware-based solutions, more secure than software vaults.

    *   **B. Token-Based Authentication (Chapter 4, Slides 30-48):**
        *   "Something you have."
        *   **Multifactor Authentication (MFA):** Combines different types of credentials (e.g., password + token).
        *   **Two-Factor Authentication (2FA):** A specific type of MFA using two types.
        *   **Specialized Devices:**
            *   **Smart Cards:** Credit-card sized, can hold info. Requires reader or contactless. Vulnerable to cloning/skimming if magnetic stripe.
            *   **Windowed Tokens (OTP - One-Time Password):**
                *   *TOTP (Time-based OTP):* Changes after a set time (e.g., 30-60s). Token and server share algorithm and time.
                *   *HOTP (HMAC-based OTP):* Event-driven, changes on specific event (e.g., PIN entry on token).
                *   *Cumbersome:* Manual entry, time-sensitive.
        *   **Smartphones (Chapter 4, Slides 40-44):**
            *   Phone call verification.
            *   SMS text message (OTP) - *Insecure: Phishable, interceptable.*
            *   Authentication App (Push notification) - *Can be targeted by malware on phone.*
        *   **Security Keys (Hardware Tokens - Chapter 4, Slides 45-48):**
            *   Dongle (USB/Lightning/NFC).
            *   Contains cryptographic info.
            *   **Attestation:** Key pair "burned" in during manufacturing, proves device model. Used to sign new credential key pairs, service verifies attestation signature.
            *   More secure than OTPs (not easily intercepted/phished).

    *   **C. Biometric Authentication (Chapter 4, Slides 49-68):**
        *   "Something you are" or "Something you do."
        *   **Physiological Biometrics:** Relates to body part functions.
            *   *Specialized Scanners:* Retinal (unique capillary patterns), Fingerprint (ridges/valleys, static/dynamic scanners), Vein (palm/finger), Gait recognition.
            *   *Standard Input Devices:* Voice Recognition (unique voice characteristics), Iris Scanner (unique iris patterns via webcam), Facial Recognition (nodal points via webcam).
            *   *Disadvantages:* Cost of specialized scanners, not foolproof (false rejections/acceptances), can be "tricked" (e.g., lifted fingerprints), privacy concerns.
        *   **Cognitive Biometrics:** Perception, thought process, understanding. (Knowledge-based).
            *   *Windows Picture Password:* Gestures (tap, line, circle) on selected points of interest on an image.
            *   *Memorable Events:* Recalling details about personal events.
            *   *Vulnerabilities:* Predictable patterns (e.g., tapping faces in picture password).

    *   **D. Behavioral Biometrics Authentication (Chapter 4, Slides 69-70):**
        *   "Something you do." Based on unique actions.
        *   **Keystroke Dynamics:** Recognizes unique typing rhythm (dwell time, flight time). Multiple samples create a user template. Convenient, no specialized hardware.

    *   **E. Authentication Security Issues & Solutions (Chapter 4, Slides 71-81):**
        *   **Issues:** Eavesdropping, host attacks, client attacks, replay, Trojan horse, denial-of-service.
        *   **Solutions (General):**
            *   **Security surrounding passwords** (covered above: salts, key stretching, password management).
            *   **Secure Authentication Technologies:**
                *   **Single Sign-On (SSO):** One set of credentials for multiple accounts. Federation for cross-organization SSO.
                *   **Authentication Services:** RADIUS, Kerberos, Directory Services, SAML.

**III. Access Control Models (Chapter 4, Slides 86-103)**

*   **A. Discretionary Access Control (DAC) (Chapter 4, Slides 87, 89-92):**
    *   Controls access based on the identity of the requestor and on access rules (authorizations).
    *   The owner of an object determines who can access it and what they can do.
    *   Termed "discretionary" because an entity with access rights can pass those rights to others.
    *   Widely implemented (e.g., UNIX file system permissions - RWX by file owners).
    *   **Access Control Matrix:**
        *   Subjects (rows) vs. Objects (columns).
        *   Each cell specifies access rights.
        *   Often sparse, can decompose into:
            *   **Access Control Lists (ACLs):** Column-focused; for each object, lists subjects and their rights.
            *   **Capability Tickets/Lists:** Row-focused; for each subject, lists objects and allowed actions.

*   **B. Mandatory Access Control (MAC) (Chapter 4, Slides 87, 93-94):**
    *   Controls access based on comparing security labels (of resources, indicating sensitivity) with security clearances (of subjects).
    *   Policy is "mandatory" because an entity cannot, by its own volition, enable another entity to access a resource.
    *   Central authority (e.g., security officer) determines access based on organizational policy.
    *   Information belongs to the organization, not individuals.
    *   Users cannot override or modify this policy.
    *   Aims to defend against Trojan horses.
    *   Generally more secure than DAC.

*   **C. Role-Based Access Control (RBAC) (Chapter 4, Slides 87, 95-97):**
    *   Access decisions are based on the roles individual users have as part of an organization.
    *   Users are assigned roles, and roles are assigned permissions to objects.
    *   Many-to-many relationship between users and roles, and roles and resources.
    *   Set of users and user-role assignments can be dynamic.
    *   Set of roles and role-permissions tend to be more static.
    *   Simplifies management compared to assigning permissions directly to many users.

*   **D. Attribute-Based Access Control (ABAC) (Chapter 4, Slides 87, 98-103):**
    *   Access is determined based on attributes (characteristics) of subjects, objects, and the environment, rather than static roles.
    *   Solves RBAC limitations in complex digital environments (cloud, IoT, mobile).
    *   **Attributes Categories:**
        *   **Subject attributes:** User's characteristics (username, age, job title, security clearance, department).
        *   **Action attributes:** The action the user wants to perform (read, write, delete, approve).
        *   **Object (Resource) attributes:** Object's characteristics (creation date, ownership, file name, data sensitivity).
        *   **Contextual (Environment) attributes:** Context of access request (time, location, device type, current threat level, transaction history).
    *   Policies define allowable operations based on these attributes (e.g., "Allow 'Managers' (subject attribute) to 'approve' (action) 'expense reports under $500' (object attribute) during 'business hours' (environment attribute)").

**IV. Physical Security (Chapter 6 - as it relates to Access Control)**

*   **Purpose:** Protect physical assets supporting information storage and processing.
*   **Two Complementary Requirements (Chapter 6, Slide 3):**
    1.  Prevent damage to physical infrastructure.
    2.  Prevent physical infrastructure misuse leading to misuse/damage of protected information.
*   **Concerns (Chapter 6, Slide 3):** Information system hardware, physical facility, support facilities, personnel.
*   **Physical Security Controls (Chapter 6, Slides 7-11):**
    *   **External Perimeter Defenses:** Restrict access to campus, building, or area.
        *   *Passive Barriers:* Fencing, signage, proper lighting.
        *   *Active Elements:* Personnel (human security guards), CCTV.
        *   *Sensors:* Motion, noise, temperature, proximity detectors to alert guards.
    *   **Internal Physical Security Controls:**
        *   **Locks:** Physical locks (keys), securing server rooms, cabinets.
        *   **Mantraps:** Air gap with two interlocking doors; only one can be open at a time to control entry/exit between nonsecure and secure areas.
        *   **Server Room/Data Center Specifics (Chapter 6, Slides 14-15):**
            *   Strong doors with strong locks (deadbolts).
            *   Key access limited to necessary personnel.
            *   Multi-layer authentication (passwords, RFID, biometrics).
            *   Server room log (manual or electronic/biometric locks).

---

### **Topic 2: About Malware (Malicious Software)**

**I. Definition and Classification (Chapter 2, Slides 3-4)**

*   **Computer Contaminant (Legal):** Any computer instructions designed to modify, damage, destroy, record, or transmit information within a computer system/network without owner's intent or permission.
*   **Malware (Malicious Software):** Software that enters a computer system without user's knowledge/consent and performs an unwanted/harmful action.
*   **Evolution:** Continuously evolving to avoid detection.
*   **Primary Action Classification:**
    1.  **Imprison:** Takes away user's freedom.
    2.  **Launch:** Infects to launch attacks on other computers.
    3.  **Snoop:** Spies on victims.
    4.  **Deceive:** Hides true intentions.
    5.  **Evade:** Helps malware/attacks evade detection.

**II. Types of Malware by Action (Chapter 2)**

*   **A. Imprison (Chapter 2, Slides 5-14)**
    *   **Ransomware:**
        *   Prevents user's device from functioning until a ransom is paid.
        *   *Blocker Ransomware:* Displays a screen, blocks access to resources (e.g., fake law enforcement warnings).
        *   *Crypto-Ransomware:* Encrypts files on the device (and potentially connected network drives/cloud storage) making them unopenable without a decryption key. Often increases ransom demand over time or threatens file deletion.
        *   Targets: Individuals, increasingly state/local governments, enterprises.

*   **B. Launch (Chapter 2, Slides 15-37)**
    *   **Virus:**
        *   **File-Based Virus:** Malicious code attached to a file. Reproduces itself on the *same computer* when the infected file is executed/opened. Requires human action (e.g., sharing infected file via email/USB) to spread to other computers.
            *   *Appender Infection:* Attaches to end of file, jump instruction at beginning redirects to virus code.
            *   *Armored File-Based Virus:* Uses techniques to avoid detection (e.g., split infection, mutation).
            *   *Payload:* Malicious action (corrupt/delete files, steal data, crash system).
        *   **Fileless Virus:** Does not attach to a file. Loads malicious code directly into RAM via **Living-Off-the-Land Binaries (LOLBins)** – native OS services/processes (e.g., PowerShell, WMI, Macros).
            *   *Advantages:* Easy to infect (malicious webpages), extensive control (via LOLBins), persistent (can write scripts to Windows Registry), difficult to detect (no file to scan, RAM-resident), difficult to defend against (disabling LOLBins cripples OS).
    *   **Worm (Network Virus):**
        *   Malicious program that uses a computer network to replicate *autonomously*.
        *   Exploits vulnerabilities in applications or OS to enter a system and then searches for other vulnerable systems on the network.
        *   Early worms slowed networks by consuming resources. Modern worms can carry damaging payloads (delete files, remote control).
    *   **Bot (Zombie):**
        *   Software allowing an infected computer to be remotely controlled by an attacker.
        *   **Botnet:** Network of bots controlled by a **bot herder**.
        *   **Command and Control (C&C):** Structure for bots to receive instructions (e.g., via websites, third-party sites, blogs, Twitter, "dead drop" Gmail drafts).
        *   *Uses:* Spamming, spreading malware, ad fraud, mining cryptocurrencies, DDoS attacks.

*   **C. Snoop (Chapter 2, Slides 38-45)**
    *   **Spyware:**
        *   Tracking software deployed without user consent/control.
        *   Secretly monitors users, collects information (personal/sensitive) via computer resources (including already installed programs).
        *   *Technologies:* Automatic download software, passive tracking, system modifying software, tracking software.
    *   **Keylogger:**
        *   Silently captures and stores keystrokes. Can also capture screen content, use webcam.
        *   *Software Keylogger:* Program installed on computer.
        *   *Hardware Keylogger:* Physical device inserted between keyboard and USB port. Resembles normal plug, hard to detect by antimalware. Requires physical access to install/retrieve.

*   **D. Deceive (Chapter 2, Slides 46-51)**
    *   **Potentially Unwanted Program (PUP):**
        *   Software user doesn't want, often bundled with other programs (user overlooks default install options) or pre-installed (bloatware).
        *   Examples: Adware obstructing content, pop-ups, search engine/homepage hijacking, toolbars, redirectors.
    *   **Trojan (Trojan Horse):**
        *   Executable program masquerading as benign but performs malicious actions.
        *   User downloads seemingly useful program (e.g., calendar) which also installs malware (e.g., data scanner, backdoor).
    *   **Remote Access Trojan (RAT):**
        *   Trojan with added functionality of giving threat agent unauthorized remote access/control over victim's computer using special communication protocols.
        *   Allows monitoring, changing settings, copying files, accessing other network computers.

*   **E. Evade (Chapter 2, Slides 52-60)**
    *   **Backdoor:**
        *   Gives access to a computer/program/service, circumventing normal security.
        *   Allows attacker to return later and bypass security.
        *   Can be legitimate (left by developers, intended for removal) or malicious.
    *   **Logic Bomb:**
        *   Computer code added to a legitimate program, lies dormant until a specific logical event triggers it (e.g., date/time, specific action).
        *   Once triggered, deletes data or performs other malicious activities.
        *   Difficult to detect before triggering (embedded in large programs, often by trusted insiders).
        *   *Examples:* UBS admin (Roger Duronio), US Army payroll (Mittesh Das), Nimesh Patel.
    *   **Rootkit:**
        *   Malware that hides its presence and other malware on the computer.
        *   Accesses "lower layers" of OS or undocumented functions to make alterations, becoming undetectable by OS/antimalware.
        *   Often installed after initial user-level access is gained (via exploit/cracked password).
        *   Can monitor traffic/keystrokes, create backdoors, alter log files, attack other machines, alter system tools.

**III. Countermeasures to Prevent Malware Attack (Chapter 2, Slides 61-76)**

*   **Developing Security Policies:** Road map for employees.
    *   *Social Engineering Awareness Policy:* Guidelines for recognizing and responding.
    *   *Server Malware Protection Policy:* Mandates anti-virus/anti-spyware on servers.
    *   *Software Installation Policy:* Requirements for installing software, minimizing risks.
    *   *Removable Media Policy:* Minimize risk from infected removable media.
*   **Implementing Security Awareness Training:** Regular training for employees.
    *   *Baseline Testing:* Assess susceptibility to phishing.
    *   *Training Users:* Interactive content on latest social engineering.
    *   *Phishing Campaigns:* Simulated attacks.
    *   *Reporting Results:* Track ROI.
    *   *Key Questions for Attachments:* Expected? Specific or generic? Doubts? (Ask tech support).
*   **Using App-Based Multi-Factor Authentication (MFA):**
    *   SMS-based MFA is weak (can be bypassed/phished).
    *   App-based MFA or hardware MFA (e.g., YubiKey) recommended.
*   **Installing Anti-Malware & Spam Filters:**
    *   On end devices and mail servers (defense in depth).
    *   Keep software up-to-date.
    *   Consider diverse vendors if using host-based and network-based antimalware.
*   **Changing Default Operating System Policies:**
    *   Improve default security settings (e.g., stronger password history, reduced password age).
    *   Network admin responsibility.
*   **Performing Routine Vulnerability Assessments:**
    *   Identify known vulnerabilities, lack of controls, misconfigurations.
    *   Scanners (e.g., Nessus) scan ports, analyze protocols, map network.
    *   Dashboards show vulnerabilities and severity.
    *   Reports include remediation plans.
    *   Implement patch management program.

---

### **Topic 3: Basics of Cryptography**

**I. Basic Definitions (Chapter 3, Slides 6-27)**

*   **Cryptography:** Practice of transforming information so it cannot be understood by unauthorized parties (secure). Achieved by "scrambling."
*   **Encryption:** Process of changing original text (plaintext) into a scrambled message (ciphertext).
*   **Decryption:** Changing ciphertext back to original plaintext.
*   **Plaintext:** Unencrypted data (input for encryption or output of decryption).
*   **Ciphertext:** Scrambled, unreadable output of encryption.
*   **Cleartext:** Unencrypted data not intended for encryption.
*   **Cipher:** Cryptographic algorithm to encrypt plaintext using procedures based on a mathematical formula.
*   **Key:** Mathematical value entered into the algorithm to produce ciphertext. Algorithms are public; keys must be secret.
*   **Elements of an Effective Cryptosystem:**
    1.  **Reversible:** Must be able to unscramble.
    2.  **Secrecy and Length of the Key:** Security depends on key secrecy/length, *not* algorithm details (unrestricted algorithm). If algorithm secrecy is required, it's a restricted algorithm.
    3.  **Substantial Cryptanalysis:** Trustworthy algorithms are thoroughly analyzed for weaknesses.
*   **Cryptanalysis:** Method to break a cipher/code, often using mathematical analysis (e.g., by NSA - "Puzzle Palace").
*   **Cipher Categories:**
    *   **Substitution Cipher:** Exchanges one character for another.
        *   *Caesar Cipher:* Shift each letter by a fixed number (Key = shift amount). Encryption: C = (P + K) mod 26. Decryption: P = (C - K) mod 26.
        *   *ROT13:* Caesar cipher with shift of 13. (A=N, B=O). *Weak, restricted algorithm.*
    *   **XOR Cipher:** Based on eXclusive OR. (Bits different -> 1, bits same -> 0). Plaintext XOR Key = Ciphertext; Ciphertext XOR Key = Plaintext.
*   **Cryptography Use Cases:**
    1.  **Confidentiality:** Ensures only authorized parties can view.
    2.  **Integrity:** Ensures information is correct and unaltered.
    3.  **Authentication:** Verifies sender's identity.
    4.  **Non-repudiation:** Inability to deny performing an action (e.g., sending an email).
    5.  **Obfuscation:** Making something obscure or unclear.
*   **Data States for Protection:**
    1.  **Data in processing (in use):** Actions being performed on data by devices.
    2.  **Data in transit (in motion):** Transmitted across a network.
    3.  **Data at rest:** Stored on electronic media.

**II. Cryptographic Algorithms Overview (Chapter 3, Slide 28)**

*   Main branches: Symmetric and Asymmetric.
*   Symmetric branches into Block Cipher and Stream Cipher.

**III. Block Ciphers vs. Stream Ciphers (Chapter 3, Slides 29-33)**

*   **Block Cipher:**
    *   Converts plaintext into ciphertext in fixed-size blocks (e.g., 64 or 128 bits).
    *   A key is applied to blocks.
    *   Uses **padding** if plaintext length isn't a multiple of block size.
    *   Example: 150-bit plaintext with 128-bit block cipher -> Block1 (128 bits), Block2 (22 bits data + 106 bits padding).
*   **Stream Cipher:**
    *   Encrypts a continuous string of binary digits (1 bit at a time).
    *   Applies time-varying transformations.
    *   Uses a **keystream** (pseudorandom number generated from key + nonce) XORed with plaintext.
    *   **Nonce:** Number Used Only Once.

**IV. Hash Algorithms (Chapter 3, Slides 34-47)**

*   **Definition:** One-way cryptographic algorithm creating a unique "digital fingerprint" (digest/hash) of data.
*   **Hashing:** The process of creating the digest.
*   **Purpose:** Primarily for comparison (verifying integrity). *Not for creating decryptable ciphertext.*
*   **One-Way:** Digest cannot be reversed to get original plaintext.
*   **Characteristics of a Secure Hash Algorithm:**
    1.  **Fixed Size:** Digest size is constant regardless of input data size.
    2.  **Unique:** Different data sets produce different digests. (Small change in data -> large change in digest). A **collision** occurs if two different inputs produce the same hash.
    3.  **Original:** Not possible to produce a data set for a predefined hash.
    4.  **Secure (Pre-image resistance):** Cannot reverse hash to find original plaintext.
*   **Use Case:** Verifying file integrity after download (compare calculated hash with website's hash).
*   **Common Hash Algorithms:**
    *   **Message Digest (MD):**
        *   Family: MD2, MD4, MD5, MD6.
        *   *MD5:* Most widely used. 32-digit hexadecimal. Used for file integrity, password encoding.
        *   *Weaknesses:* Vulnerable to collisions (two files having same MD5 code). No longer considered secure.
    *   **Secure Hash Algorithm (SHA):**
        *   *SHA-1:* Developed 1993, no longer suitable.
        *   *SHA-2:* Variations (SHA-256, SHA-384, SHA-512 – number indicates digest length in bits). Currently considered secure.
        *   *SHA-3:* New standard (2015), dissimilar to previous SHA to prevent building on prior compromises.
    *   **Hashing Message Authentication Code (HMAC):**
        *   Detects intentional alterations in a message by incorporating a secret key into the hashing process.
        *   A keyed cryptographic hash function.
        *   *Process:* Sender and recipient pre-share a secret key. Sender hashes message, then XORs hash with key (or a more complex combination). Recipient does the same. If results match, integrity and authenticity (of key holder) are verified. Interceptor without key cannot produce matching HMAC.

**V. Symmetric Encryption Principles (Chapter 3, Slides 48-58)**

*   Uses the **same key** to encrypt and decrypt data.
*   Also called **private key cryptography** (key must be kept private).
*   Strength depends on key size and secrecy.
*   Relatively fast.
*   **Weaknesses:**
    *   Key sharing: Difficult to share key securely over an unsecured network.
    *   No inherent process for authentication or non-repudiation.
*   **Common Symmetric Algorithms:**
    *   **Data Encryption Standard (DES):**
        *   Early algorithm (IBM, 1970s). Block cipher (64-bit blocks).
        *   Key size: 56 bits (+8 parity). *Too small for modern brute-force attacks, not secure.*
    *   **Triple DES (3DES):**
        *   Replaced DES. Uses three rounds of DES encryption.
        *   Ciphertext of one round is input for the next.
        *   More secure versions use different keys per round.
        *   Slower performance. No longer considered most secure.
    *   **Advanced Encryption Standard (AES):**
        *   Current standard, replaced DES. Most widely used symmetric key algorithm.
        *   Block cipher (128-bit blocks; can be 192/256).
        *   Key sizes: 128, 192, or 256 bits.
        *   AES-256 considered strong enough for military TOP SECRET data.
        *   Highly resistant to brute-force.
    *   **Rivest Cipher (RC):**
        *   *RC4:* Stream cipher, keys up to 128 bits. Used in SSL/TLS, early Wi-Fi. *Considered insecure today.*
        *   *RC5:* Block cipher, variable block size/key length/rounds. Still considered secure with suitable parameters.
        *   *RC6:* Fixed 128-bit block size.
        *   RC5/RC6 not widely used (patented).

**VI. Asymmetric Encryption or Public Key Cryptography (Chapter 3, Slides 59-82)**

*   Uses **two keys:** a public key and a private key, mathematically related.
*   **Public Key:** Known to everyone, freely distributed.
*   **Private Key:** Known only to the owner, kept confidential.
*   **Process:** To send a secure message to Alice, Bob encrypts with Alice's *public key*. Alice decrypts with her *private key*.
*   **Key Pairs:** Essential.
*   **Both Directions:** Encryption with public key -> decryption with private key. Encryption with private key -> decryption with public key (used for digital signatures).
*   **Common Asymmetric Algorithms:**
    *   **RSA (Rivest-Shamir-Adleman):**
        *   Published 1977. Block cipher (e.g., 1024 bits).
        *   *Basis:* Difficulty of factoring large prime numbers.
        *   *Key Generation:*
            1.  Select two large prime numbers, `p` and `q`.
            2.  Compute `n = p * q`.
            3.  Compute `m (φ(n)) = (p-1) * (q-1)`.
            4.  Choose `e` (public exponent) < `n`, `e` coprime to `m`.
            5.  Determine `d` (private exponent) such that `(e*d - 1)` is divisible by `m` (i.e., `e*d ≡ 1 (mod m)`).
            6.  Public Key: `(n, e)`. Private Key: `(n, d)`. `p`, `q` discarded.
        *   *Usage:* Web browsers, email, VPNs, TLS handshakes.
    *   **Elliptic Curve Cryptography (ECC):**
        *   Uses sloping curves (set of points satisfying an elliptic curve equation) instead of large prime factoring.
        *   Operations involve adding points on the curve.
        *   Offers same security level as RSA with much smaller key sizes.
            *   e.g., 228-bit ECC key security >> 228-bit RSA key.
            *   Breaking 228-bit ECC requires more energy than boiling all water on Earth.
        *   *Advantages:* Faster computations, lower power consumption (good for mobile devices).
        *   *Usage:* US gov internal comms, Tor, Bitcoin ownership, modern OS/browsers.
    *   **Digital Signature Algorithm (DSA):**
        *   Used for creating **digital signatures** (electronic verification of sender).
        *   Provides:
            1.  **Sender Verification (Authentication):** Confirms message origin.
            2.  **Non-repudiation:** Sender cannot deny sending.
            3.  **Message Integrity:** Proves message hasn't been altered.
        *   *Process (Bob sends signed message to Alice):*
            1.  Bob creates message digest (hash).
            2.  Bob encrypts digest with *his private key* (this is the digital signature).
            3.  Bob sends memo + digital signature to Alice.
            4.  Alice decrypts signature with *Bob's public key* to get Bob's original digest. (If fails, not from Bob).
            5.  Alice hashes the received memo herself and compares her hash with decrypted digest. (If match, integrity and authenticity proven).
        *   *Note:* DSA only signs; does not encrypt the message itself. For confidentiality, message would also need encryption with Alice's public key.
        *   US federal government standard (part of DSS).
    *   **Diffie-Hellman Key Exchange:**
        *   Method for two parties to securely exchange a symmetric key over an insecure channel *without prior shared secrets*.
        *   *Process (Alice and Bob):*
            1.  Agree on large prime `p` and related integer `g` (generator) (public).
            2.  Alice chooses secret `a`, Bob chooses secret `b`.
            3.  Alice calculates her public key `A = g^a mod p`.
            4.  Bob calculates his public key `B = g^b mod p`.
            5.  They exchange `A` and `B` publicly.
            6.  Alice calculates shared secret `s = B^a mod p = (g^b)^a mod p`.
            7.  Bob calculates shared secret `s = A^b mod p = (g^a)^b mod p`.
            8.  Both arrive at the same secret key `s = g^(ab) mod p`.

**VII. Cryptographic Attacks (Chapter 3, Slides 83-85)**

*   Algorithms undergo rigorous review, but implementation weaknesses can be exploited.
*   **Algorithm Attacks:**
    *   **Known Ciphertext Attacks:** Analyzing ciphertexts for patterns to reveal plaintext/keys.
    *   **Downgrade Attack:** Forcing systems to use older, less secure modes during negotiation.
    *   **Misconfigurations:** Incorrect choices of crypto options (e.g., using DES/SHA-1 instead of AES/SHA-256).
*   **Collision Attacks (on Hash Algorithms):**
    *   Exploits weakness where two different inputs produce the same hash digest.
    *   **Birthday Attack:** Based on "birthday paradox"; easier to find *any* two inputs that collide than a specific input for a hash.

**VIII. Quantum Cryptographic Defenses (Chapter 3, Slide 86)**

*   **Quantum Computers:** Use qubits (0 and 1 simultaneously). Potentially faster/more efficient. Delicate, prone to faults, no commercial-grade yet.
*   **Quantum Cryptography:** Exploits quantum mechanics for enhanced security (e.g., detecting eavesdroppers).
*   **Threat:** Quantum computers can quickly factor large numbers, rendering current asymmetric algorithms (like RSA) useless.
*   **Post-Quantum Cryptography (PQC):** Developing encryption resistant to quantum computer attacks. Still in future for widespread implementation.

**IX. Application of Cryptography (Chapter 3, Slides 87-92)**

*   **A. Software-based Cryptography:**
    *   **File and File System Cryptography:**
        *   Encrypt/decrypt individual files (e.g., GNuPG).
        *   OS features for folder encryption (e.g., Microsoft's EFS).
    *   **Full Disk Encryption (FDE):**
        *   Protects entire disks (e.g., Microsoft's BitLocker). Prevents unauthorized access even if drive removed.
*   **B. Hardware Encryption:**
    *   **USB Device Encryption:** Prevents data leakage from lost/stolen USBs. Requires password, auto-encrypts, tamper-resistant.
    *   **Self-Encrypting Drives (SED):** Automatically encrypts all data on drive. Authentication at startup. Cryptographic erase on auth fail.
    *   **Hardware Security Module (HSM):** Removable external device. Random number generator, key storage, accelerated encryption, malware-proof.
    *   **Trusted Platform Module (TPM):** Chip on motherboard. Cryptographic services, true random number generator, asymmetric encryption support, measures key components at startup, recovery password if drive moved.
*   **C. Blockchain:**
    *   Shared, tamper-evident, distributed ledger.
    *   Transactions recorded once, preventing alterations.
    *   All parties must agree (consensus) before new transaction added.
    *   Relies on cryptographic hash algorithms (e.g., SHA-256) for security and immutability (linking blocks via hashes).

---

## Part 2: Flashcards

---

(All questions first, then all answers)

**Flashcard Questions: Access Control**

1.  Q: What is Authentication in the context of access control?
2.  Q: What is Authorization in the context of access control?
3.  Q: What are the two steps in the authentication process?
4.  Q: Name the four general means of authenticating user identity.
5.  Q: What is a "Pass the Hash" attack?
6.  Q: What is "Password Spraying"?
7.  Q: What is a "Rainbow Table"?
8.  Q: What is the purpose of "salts" in password hashing?
9.  Q: What is "key stretching" in password security?
10. Q: What is Multifactor Authentication (MFA)?
11. Q: What is a TOTP?
12. Q: What is a HOTP?
13. Q: What is "attestation" in the context of security keys?
14. Q: Differentiate between physiological and cognitive biometrics.
15. Q: What is keystroke dynamics?
16. Q: What is Discretionary Access Control (DAC)?
17. Q: What is an Access Control List (ACL)?
18. Q: What is a Capability List/Ticket?
19. Q: What is Mandatory Access Control (MAC)?
20. Q: What is Role-Based Access Control (RBAC)?
21. Q: What is Attribute-Based Access Control (ABAC)?
22. Q: Name two types of external perimeter defenses for physical security.
23. Q: What is a mantrap?

**Flashcard Questions: About Malware**

24. Q: What is malware?
25. Q: What is Ransomware?
26. Q: Differentiate between blocker ransomware and crypto-ransomware.
27. Q: What is a file-based virus? How does it spread?
28. Q: What is a fileless virus? What does it use to operate?
29. Q: What are LOLBins? Give an example.
30. Q: What is a Worm? How does it differ from a virus in spreading?
31. Q: What is a Bot/Zombie in malware terms?
32. Q: What is a Botnet?
33. Q: What is Spyware?
34. Q: What is a Keylogger? Name two forms.
35. Q: What is a Potentially Unwanted Program (PUP)?
36. Q: What is a Trojan horse in malware?
37. Q: What is a Remote Access Trojan (RAT)?
38. Q: What is a Backdoor in the context of malware?
39. Q: What is a Logic Bomb?
40. Q: What is a Rootkit? How does it achieve stealth?
41. Q: Name three general countermeasures against malware.

**Flashcard Questions: Basics of Cryptography**

42. Q: Define Cryptography.
43. Q: Differentiate between encryption and decryption.
44. Q: What is a cipher? What is a key?
45. Q: What is the difference between a restricted and an unrestricted algorithm?
46. Q: What is a substitution cipher? Give an example.
47. Q: Name three use cases for cryptography.
48. Q: Differentiate between a block cipher and a stream cipher.
49. Q: What is a hash algorithm?
50. Q: What are the four key characteristics of a secure hash algorithm?
51. Q: What is a "collision" in hashing?
52. Q: What is HMAC? What is its primary purpose?
53. Q: What is symmetric encryption? What is another name for it?
54. Q: What is the main weakness of symmetric encryption?
55. Q: Name a widely used symmetric encryption algorithm.
56. Q: What is asymmetric encryption? What is another name for it?
57. Q: How are the two keys used in asymmetric encryption for sending a confidential message?
58. Q: Name a widely used asymmetric encryption algorithm and its mathematical basis.
59. Q: What is a digital signature? What three things can it prove?
60. Q: How is a digital signature created and verified using asymmetric keys?
61. Q: What is the Diffie-Hellman Key Exchange used for?
62. Q: What is a downgrade attack in cryptography?
63. Q: What is an HSM?
64. Q: What is a TPM?
65. Q: How does blockchain ensure security and immutability?

---

**Flashcard Answers**

*Access Control Answers:*

1.  A: Verifying that the credentials of a user or system entity are valid.
2.  A: Granting a right or permission to a system entity to access a system resource.
3.  A: 1. Identification step (presenting an identifier, e.g., User ID). 2. Verification step (presenting corroborating info, e.g., Password).
4.  A: Something you know, something you have, something you are, something you do.
5.  A: An attack where an attacker uses a stolen password hash (digest) to impersonate a legitimate user without needing the plaintext password.
6.  A: An attack method where an attacker tries a small number of common passwords against many different user accounts.
7.  A: A pre-computed table used to reverse cryptographic hash functions, typically for cracking password hashes. It stores hash values and their corresponding plaintexts.
8.  A: To ensure that identical passwords hash to different values, making pre-computed tables (like rainbow tables) ineffective and slowing down brute-force attacks.
9.  A: A technique to make password hashing slower by repeatedly applying a hash function or using computationally intensive algorithms, increasing the effort needed for attackers to crack passwords.
10. A: An authentication method that requires the user to provide two or more verification factors to gain access to a resource.
11. A: Time-based One-Time Password; an OTP that is valid only for a short period.
12. A: HMAC-based One-Time Password; an OTP that changes based on a specific event, like a button press or counter.
13. A: A process where a security key cryptographically proves its genuineness and model to a service, often using a built-in key pair from manufacturing.
14. A: Physiological biometrics relate to physical characteristics of the body (e.g., fingerprint, face). Cognitive biometrics relate to perception, thought processes, or memory (e.g., picture password, memorable events).
15. A: A behavioral biometric that identifies users based on their unique typing rhythm, including speed, dwell time (key press duration), and flight time (time between key presses).
16. A: An access control model where the owner of a resource (object) determines who can access it and what permissions they have.
17. A: A list associated with an object that specifies which subjects are allowed to access it and what operations they can perform.
18. A: A token/list associated with a subject that specifies which objects the subject can access and what operations they can perform on those objects.
19. A: An access control model where access decisions are based on security labels (of objects) and security clearances (of subjects), enforced by a central authority.
20. A: An access control model where access permissions are assigned to roles, and users are assigned to roles, thereby inheriting permissions.
21. A: An access control model where access rights are granted based on policies that evaluate attributes of the subject, object, action, and environment.
22. A: Fencing, security guards, CCTV, proper lighting, signage. (Any two)
23. A: A physical security control with two interlocking doors, where only one can be open at a time, used to control access to a secure area.

*About Malware Answers:*

24. A: Software designed to enter a computer system without the user's knowledge or consent to perform an unwanted and harmful action.
25. A: Malware that prevents a user's device from functioning properly or encrypts their files, demanding a payment (ransom) for restoration.
26. A: Blocker ransomware displays a screen preventing access to the system. Crypto-ransomware encrypts user files, making them inaccessible.
27. A: Malicious code attached to a file. It replicates on the same computer when the file is executed and spreads to other computers via human action (e.g., sharing the file).
28. A: Malware that does not rely on traditional executable files. It operates in memory, often leveraging legitimate system tools (LOLBins).
29. A: Living-Off-the-Land Binaries. Legitimate system tools misused by malware. Example: PowerShell, WMI, Macros.
30. A: A self-replicating malicious program that spreads across networks, typically by exploiting vulnerabilities, without human intervention. Viruses need human action to spread between computers.
31. A: An infected computer that is under the remote control of an attacker.
32. A: A network of compromised computers (bots) controlled by a bot herder, often used for coordinated attacks or illicit activities.
33. A: Malware that secretly monitors user activity and collects information without their consent.
34. A: Malware that records a user's keystrokes. Forms: Software (program) and Hardware (physical device).
35. A: Software that a user may not want, often bundled with other downloads or pre-installed, and can include adware or toolbars.
36. A: Malicious software disguised as a legitimate or useful program.
37. A: A type of Trojan that provides an attacker with unauthorized remote administrative control over an infected computer.
38. A: A hidden method of bypassing normal authentication or security controls to gain access to a system or data.
39. A: Malicious code embedded in a legitimate program that remains dormant until a specific condition or event (e.g., a date/time) triggers it to execute a harmful action.
40. A: Malware designed to hide its presence and other malware on a system by modifying the operating system at a low level.
41. A: Developing Security Policies, Implementing Security Awareness Training, Using MFA, Installing Anti-Malware, Changing Default OS Policies, Vulnerability Assessments. (Any three)

*Basics of Cryptography Answers:*

42. A: The practice and study of techniques for secure communication in the presence of third parties (adversaries), primarily by transforming information to make it unintelligible.
43. A: Encryption transforms plaintext into ciphertext. Decryption transforms ciphertext back into plaintext.
44. A: A cipher is the cryptographic algorithm used for encryption/decryption. A key is a piece of information (a parameter) that determines the functional output of a cryptographic algorithm.
45. A: A restricted algorithm's security relies on keeping the algorithm itself secret. An unrestricted algorithm's security relies on the secrecy of the key, even if the algorithm is public.
46. A: A cipher where each character in the plaintext is replaced by another character. Example: Caesar cipher, ROT13.
47. A: Confidentiality, Integrity, Authentication, Non-repudiation, Obfuscation. (Any three)
48. A: A block cipher encrypts data in fixed-size blocks (e.g., 128 bits). A stream cipher encrypts data bit by bit or byte by byte, often using a keystream.
49. A: A one-way function that takes an arbitrary-sized input and produces a fixed-size string of characters, the hash value (or digest).
50. A: 1. Fixed Size (output is always same length). 2. Unique (different inputs rarely produce same hash - collision resistance). 3. Original (Pre-image resistance - hard to find input for a given hash). 4. Secure (Second pre-image resistance - hard to find another input for a given input's hash). *(Note: Slides use slightly different phrasing, capture the essence)*
51. A: When two different inputs to a hash function produce the exact same hash output.
52. A: Hashing Message Authentication Code. It combines a cryptographic hash function with a secret key to verify both data integrity and authenticity of a message.
53. A: A type of encryption where the same key is used for both encryption and decryption. Also known as private key cryptography.
54. A: The secure distribution and management of the shared secret key.
55. A: AES (Advanced Encryption Standard). Others: DES, 3DES, RC family.
56. A: A type of encryption that uses a pair of keys: a public key for encryption and a private key for decryption (or vice-versa for signatures). Also known as public key cryptography.
57. A: To send a confidential message to Alice, Bob encrypts the message using Alice's *public key*. Alice then decrypts it using her *private key*.
58. A: RSA (Rivest-Shamir-Adleman). Based on the difficulty of factoring the product of two large prime numbers. ECC is another, based on elliptic curve discrete logarithm problem.
59. A: An electronic verification of a sender's identity and the integrity of a message. It can prove: 1. Sender authenticity, 2. Message integrity (unaltered), 3. Non-repudiation by the sender.
60. A: Created by hashing the message and then encrypting the hash with the sender's *private key*. Verified by decrypting the signature with the sender's *public key* and comparing the result with a freshly computed hash of the received message.
61. A: To allow two parties to establish a shared secret key over an insecure communication channel, which can then be used for symmetric encryption.
62. A: An attack where an attacker forces a system to abandon a higher-security mode of operation in favor of an older, less secure mode.
63. A: Hardware Security Module. A physical device that safeguards and manages digital keys, performs encryption and decryption, and provides strong authentication.
64. A: Trusted Platform Module. A dedicated microcontroller designed to secure hardware by integrating cryptographic keys into devices.
65. A: Through cryptographic hashes that link each block to the previous one (forming a chain) and a distributed consensus mechanism, making it extremely difficult to alter past transactions.

---

## Part 3: Mini-Tests

---

**Mini-Test 1: Access Control (15 minutes)**

1.  **Multiple Choice (Choose one):** Which access control model is primarily concerned with data ownership, allowing the owner of a resource to decide who can access it?
    a)  MAC (Mandatory Access Control)
    b)  RBAC (Role-Based Access Control)
    c)  DAC (Discretionary Access Control)
    d)  ABAC (Attribute-Based Access Control)

2.  **Short Answer:** Briefly explain the difference between authentication and authorization.

3.  **Scenario:** A company wants to ensure that only active employees in the "Finance Department" can access "Payroll System" on "Weekdays" between "9 AM and 5 PM" using a "company-issued laptop". Which access control model would be most suitable for implementing such a granular policy and why? Name two attribute categories involved.

4.  **True/False (Justify):** Using salts in password hashing prevents rainbow table attacks entirely.

5.  **List:** Name three different "means of authentication" categories.

**Mini-Test 1: Answers**

1.  c) DAC (Discretionary Access Control)
2.  Authentication is the process of verifying a user's identity (proving they are who they claim to be, e.g., via password). Authorization is the process of granting or denying that verified user specific permissions to access resources based on their identity or role.
3.  ABAC (Attribute-Based Access Control) would be most suitable because it allows for fine-grained policies based on multiple characteristics.
    *   Attributes involved could be: Subject attributes (department: Finance, status: active employee), Resource attributes (system: Payroll System), Environment attributes (time: 9AM-5PM, day: Weekdays, device type: company-issued laptop).
4.  True (mostly). Salts make pre-computed rainbow tables for common passwords ineffective because the hash is now for "password + salt" not just "password". Each user's salt is different, so a unique rainbow table would be needed for each user's specific salt, which is computationally infeasible for attackers if salts are unique and long enough. *(A nuanced answer acknowledging the significant difficulty it creates is good)*
5.  Something you know, Something you have, Something you are/do.

---

**Mini-Test 2: About Malware (15 minutes)**

1.  **Match the Malware type to its primary characteristic:**
    *   A. Ransomware 1. Hides its presence by modifying OS.
    *   B. Worm 2. Spies on user activity.
    *   C. Rootkit 3. Encrypts files demanding payment.
    *   D. Spyware 4. Self-replicates across networks autonomously.

2.  **Short Answer:** What are LOLBins and why are they effective for fileless malware?

3.  **Scenario:** An employee receives an email with a link to download a "Company Annual Report PDF". After clicking and opening the PDF, their computer starts acting erratically, and sensitive company files are later found leaked online. What type of malware (from the "Deceive" category) was likely involved in the PDF and what additional malware might it have installed?

4.  **List:** Name three common countermeasures organizations can implement to prevent malware attacks.

**Mini-Test 2: Answers**

1.  A-3, B-4, C-1, D-2
2.  LOLBins (Living-Off-the-Land Binaries) are legitimate operating system tools and processes. They are effective for fileless malware because their activity is often trusted by security software, allowing the malware to operate without dropping new malicious files that could be detected by traditional antivirus signatures.
3.  The PDF likely contained a **Trojan**. It masqueraded as a legitimate report but carried a malicious payload. This Trojan could have installed other malware, such as a **RAT (Remote Access Trojan)** or **Spyware** to exfiltrate the sensitive files.
4.  Implementing Security Awareness Training, Installing Anti-Malware & Spam Filters, Performing Routine Vulnerability Assessments (or other valid ones like MFA, security policies, OS hardening).

---

**Mini-Test 3: Basics of Cryptography (15 minutes)**

1.  **Multiple Choice (Choose one):** Which of the following ensures that a sender cannot deny having sent a message?
    a)  Confidentiality
    b)  Integrity
    c)  Authentication
    d)  Non-repudiation

2.  **Short Answer:** What is the fundamental difference between symmetric and asymmetric encryption regarding keys?

3.  **Explain:** How does a digital signature verify both the authenticity of the sender and the integrity of the message?

4.  **True/False (Justify):** A hash function is designed to be easily reversible to obtain the original plaintext.

5.  **Fill in the blanks:** AES is a type of ______ cipher, while RC4 is a type of ______ cipher. RSA relies on the difficulty of ______.

**Mini-Test 3: Answers**

1.  d) Non-repudiation
2.  Symmetric encryption uses the same single key for both encryption and decryption. Asymmetric encryption uses a pair of keys: a public key and a private key.
3.  A digital signature is created by hashing the message and then encrypting the hash with the sender's private key.
    *   **Authenticity:** Only the sender's corresponding public key can decrypt the signature. If decryption is successful, it proves the signature (and thus the message, by association with the hash) came from the owner of that private key.
    *   **Integrity:** The recipient re-hashes the received message. If this new hash matches the decrypted hash from the signature, it proves the message has not been altered since it was signed.
4.  False. Hash functions are designed to be one-way; it should be computationally infeasible to reverse the hash to obtain the original plaintext.
5.  AES is a type of **block** cipher, while RC4 is a type of **stream** cipher. RSA relies on the difficulty of **factoring large prime numbers**.

---

## Part 4: Mock Exam Questions

---

**Section A: Short Answer & Concepts (Attempt ALL questions)**

1.  **(a)** Define "Computer Security" based on the NIST definition provided in your course materials (Chapter 1, Slide 4). (3 marks)

** (b)** Briefly explain the six principles of security: Confidentiality, Authentication, Integrity, Non-Repudiation, Access Control, and Availability. (12 marks)

2.  With reference to Physical Security (Chapter 6):
    **(a)** Describe two complementary requirements of physical security. (4 marks)
    
    **(b)** Explain how a mantrap enhances physical access control. (3 marks)
    
    **(c)** List three types of sensors used in external perimeter defenses and briefly state their purpose. (3 marks)

4.  **Malware (Chapter 2):**
    **(a)** Differentiate between a file-based virus and a fileless virus in terms of their operation and detection challenges. (6 marks)
    
    **(b)** Explain the primary function of a Rootkit and why it is particularly difficult to detect. (4 marks)

**Section B: Scenarios & Application (Attempt ALL questions)**

4.  **Cryptography (Chapter 3):**
    **(a)** A company needs to ensure that financial transaction messages sent between its branches are not tampered with during transmission and that the sending branch can be verified.
    
        **(i)** Which cryptographic technique should they primarily use to achieve these goals? (1 mark)
    
        **(ii)** Briefly outline the steps involved in applying this technique for a message sent from Branch A to Branch B. (6 marks)
    
    **(b)** Compare and contrast symmetric (e.g., AES) and asymmetric (e.g., RSA) encryption, highlighting one key advantage and one key disadvantage of each. (8 marks)
    
    **(c)** Briefly explain the purpose of the Diffie-Hellman key exchange algorithm. (3 marks)

6.  **Access Control & Authentication (Chapter 4):**
    **(a)** An organization is designing its access control system. They are considering DAC, MAC, and RBAC.
    
        **(i)** For a highly sensitive government database where data classification and user clearances are paramount, which model would be most appropriate and why? (3 marks)
    
        **(ii)** For managing permissions for a large number of users in a typical corporate environment with defined job functions (e.g., Sales, HR, IT Support), which model would be most efficient and why? (3 marks)
    
    **(b)** An online banking application requires users to authenticate.
    
        **(i)** Propose a robust multifactor authentication (MFA) setup for this application, naming at least two distinct factors and the specific technologies/methods for each. (4 marks)
    
        **(ii)** Explain one common attack against password authentication and one countermeasure to mitigate it (other than MFA). (5 marks)

---

**Mock Exam Model Answers (Key Points)**

**Section A:**

1.  **(a) Computer Security (NIST):** The protection afforded to an automated information system to attain the applicable objectives of preserving the integrity, availability, and confidentiality of information system resources (includes hardware, software, firmware, information/data, and telecommunications).

    **(b) Six Principles:**
    *   **Confidentiality:** Only sender and intended recipient(s) access message content. Prevents unauthorized disclosure.
    *   **Authentication:** Establishes proof of identities. Ensures origin of message/document is correctly identified.
    *   **Integrity:** Ensures message content is not changed during transmission or storage.
    *   **Non-Repudiation:** Prevents a user from denying they sent a message or performed an action.
    *   **Access Control:** Determines who can access what resources and perform what actions.
    *   **Availability:** Resources are accessible to authorized parties at all times.

2.  **(a) Physical Security Requirements:**
    *   Prevent damage to physical infrastructure.
    *   Prevent physical infrastructure misuse that leads to misuse/damage of protected information.
    
    **(b) Mantrap:** An air gap with two interlocking doors. Only one door can be open at any time, controlling passage between a nonsecure and a secure area, preventing tailgating and ensuring only authorized individuals pass through one at a time.

    **(c) Sensors:**
    *   Motion detectors: Detect movement in a restricted area.
    *   Noise detectors: Detect suspicious noises.
    *   Proximity detectors: Detect presence of an object when it enters the sensor's field.

4.  **(a) File-based vs. Fileless Virus:**
    *   **File-based:** Attaches to executable files. Spreads when infected file is shared and run. Replicates on the same computer. *Detection:* Antivirus can scan files for known signatures.
    *   **Fileless:** Does not write to disk as a traditional file. Resides in memory (RAM). Uses legitimate system tools (LOLBins) to execute. Spreads often via exploits or scripts. *Detection Challenge:* No file to scan, uses legitimate tools which are harder to flag as malicious, RAM analysis needed.
    **(b) Rootkit:** Hides its presence and other malware. Accesses low levels of the OS or uses undocumented functions to modify system behavior, making itself and other malware invisible to the OS and standard antimalware tools.

**Section B:**

4.  **(a) (i) Cryptographic Technique:** Digital Signatures (using asymmetric cryptography and hashing).

    **(a) (ii) Steps for Digital Signature (Branch A to Branch B):**
    1.  Branch A creates the financial transaction message.
    2.  Branch A calculates a hash (digest) of the message.
    3.  Branch A encrypts this hash with its *own private key*. This encrypted hash is the digital signature.
    4.  Branch A sends the original message and the digital signature to Branch B.
    5.  Branch B receives the message and the signature.
    6.  Branch B decrypts the signature using Branch A's *public key* to retrieve the original hash.
    7.  Branch B calculates a new hash of the received message.
    8.  Branch B compares the new hash with the decrypted hash. If they match, the message is authentic (from Branch A) and has integrity (not tampered).
    
    **(b) Symmetric vs. Asymmetric Encryption:**
    *   **Symmetric (e.g., AES):**
        *   *Key Usage:* Uses a single, shared secret key for both encryption and decryption.
        *   *Advantage:* Fast encryption/decryption speed, computationally less intensive.
        *   *Disadvantage:* Secure key distribution is a major challenge; difficult to share the secret key securely.
    *   **Asymmetric (e.g., RSA):**
        *   *Key Usage:* Uses a pair of keys: a public key (for encryption or signature verification) and a private key (for decryption or signature creation).
        *   *Advantage:* Solves the key distribution problem (public key can be shared openly); enables digital signatures for authentication and non-repudiation.
        *   *Disadvantage:* Slower than symmetric encryption, computationally more intensive.
   
    **(c) Diffie-Hellman:** Enables two parties, without any prior shared secret, to establish a shared secret key over an insecure communication channel. This shared secret key can then be used for symmetric encryption.

5.  **(a) (i) Highly Sensitive Government Database:** MAC (Mandatory Access Control). Because MAC enforces a centrally managed security policy based on classifications (labels on data) and clearances (for users), which cannot be overridden by users. This is ideal for environments where information control is strict and based on sensitivity levels.

    **(a) (ii) Corporate Environment with Job Functions:** RBAC (Role-Based Access Control). Because permissions are assigned to roles (e.g., Sales_Manager, HR_Specialist) and users are assigned to these roles. This simplifies administration as permissions are managed at the role level, not for individual users, which is efficient for large organizations with defined job responsibilities.

    **(b) (i) MFA for Online Banking:**
    *   **Factor 1 (Something you know):** Strong, unique password.
    *   **Factor 2 (Something you have):**
        *   *Method 1:* Time-based One-Time Password (TOTP) generated by an authenticator app on a smartphone (e.g., Google Authenticator, Authy).
        *   *Method 2 (More Secure):* A hardware security key (e.g., YubiKey) that performs cryptographic operations and requires physical presence (USB/NFC).
    
    **(b) (ii) Password Attack & Countermeasure:**
    *   **Attack:** Dictionary Attack (or Brute Force Attack). Attacker tries common words (or all combinations) as passwords, hashing them and comparing against a stolen password hash file.
    *   **Countermeasure (other than MFA):**
        *   **Salting Passwords:** Adding a unique random string (salt) to each password before hashing. This ensures that even if two users have the same password, their stored hashes will be different, rendering pre-computed hash tables (like rainbow tables) ineffective against salted hashes.
        *   **Key Stretching:** Using algorithms like bcrypt or Argon2 that deliberately slow down the hashing process, making it computationally expensive and time-consuming for attackers to try many password guesses.

---

This should give you a solid foundation for your revision. Remember to go back to your slides to cross-reference and get more detailed examples! Good luck!




Okay, let's focus on these two specific areas for your BAIT1093 revision:

1.  **Security Principles Usage:** This involves understanding how the core security principles (Confidentiality, Integrity, Availability, Authentication, Non-repudiation, Access Control) are applied or what happens when they are compromised. We'll draw examples from various chapters.
2.  **Asymmetric Encryption or Public Key Cryptography (in calculation):** This will focus on the mathematical aspects of RSA and Diffie-Hellman as shown in Chapter 3.

---

## Part 1: Revision Notes

---

### **Topic 1: Security Principles Usage**

(Primarily from Chapter 1, Slides 5-13, but applied across various contexts)

**Core Security Principles:**

*   **Confidentiality:**
    *   **Definition:** Ensuring that information is not disclosed to unauthorized individuals, entities, or processes. Only the sender and intended recipient(s) should access the content.
    *   **Compromise Example (Chapter 1, Slide 6):** An unauthorized person intercepts and reads a secret message (e.g., through network sniffing if data is unencrypted).
    *   **Usage/Application:**
        *   Encrypting data at rest (e.g., BitLocker - Chapter 3, Slide 87) and in transit (e.g., HTTPS/SSL/TLS - Chapter 5, Slide 45).
        *   Access controls limiting who can view sensitive files.
        *   Physical security preventing unauthorized viewing of screens or documents.
*   **Integrity:**
    *   **Definition:** Maintaining the consistency, accuracy, and trustworthiness of data over its entire lifecycle. Data cannot be improperly modified by unauthorized parties.
    *   **Compromise Example (Chapter 1, Slide 8):** The contents of a message are changed during transmission from sender to receiver (modification attack). An employee making an unauthorized modification to a customer's bank balance (Exam Question).
    *   **Usage/Application:**
        *   **Hash functions (MD5, SHA - Chapter 3, Slides 34-44):** To verify data hasn't changed.
        *   **Digital signatures (Chapter 3, Slides 70-77):** To ensure message integrity and authenticity.
        *   **HMAC (Chapter 3, Slides 45-47):** To provide keyed integrity for messages.
        *   Access controls preventing unauthorized writes or deletions.
        *   Version control systems.
*   **Availability:**
    *   **Definition:** Ensuring that systems and data are accessible and usable upon demand by an authorized entity.
    *   **Compromise Example (Chapter 1, Slide 13):** A Denial-of-Service (DoS) attack makes a web server or network resource unavailable to legitimate users (interruption attack). Ransomware encrypting files makes them unavailable (Chapter 2).
    *   **Usage/Application:**
        *   Redundancy (e.g., RAID, backup servers, multiple internet connections - Chapter 6, Slide 16 on server redundancy).
        *   DoS protection mechanisms (e.g., firewalls, traffic scrubbing services).
        *   Regular backups and disaster recovery plans.
        *   Patch management to prevent exploits that could cause system crashes.
*   **Authentication:**
    *   **Definition:** Verifying the identity of a user, process, or device; proving that an entity is who or what it claims to be.
    *   **Compromise Example (Chapter 1, Slide 7):** An attacker impersonates a legitimate user by stealing their credentials (fabrication attack).
    *   **Usage/Application:**
        *   Passwords, PINs (Chapter 4).
        *   Tokens, Smartcards, Security Keys (Chapter 4).
        *   Biometrics (fingerprint, facial recognition - Chapter 4).
        *   Digital certificates (Chapter 5, Slide 43).
        *   Multi-Factor Authentication (MFA).
*   **Non-Repudiation:**
    *   **Definition:** Ensuring that a party cannot deny the authenticity of their signature on a document or the sending of a message that they originated. Provides proof of origin and integrity.
    *   **Compromise Example (Chapter 1, Slide 9):** A user sends a malicious email and then falsely claims their account was hacked and they didn't send it.
    *   **Usage/Application:**
        *   **Digital signatures (Chapter 3, Slides 70-77):** Asymmetrically encrypting a hash of a message with a private key links the sender to the message.
        *   Audit logs that track user actions with strong authentication.
*   **Access Control:**
    *   **Definition:** The ability to control who can access specific resources (data, systems, physical locations) and what actions they are permitted to perform (read, write, execute).
    *   **Compromise Example (Chapter 1, Slide 10 related to role/rule management):** A user with basic privileges manages to gain administrative access to a system or a sensitive file due to misconfigured permissions.
    *   **Usage/Application:**
        *   Access Control Lists (ACLs), Capability Lists (Chapter 4, Slide 91).
        *   DAC, MAC, RBAC, ABAC models (Chapter 4, Slides 87-103).
        *   Firewall rules (Chapter 5, Slides 28-37).
        *   Physical access controls like key cards, locks, mantraps (Chapter 6).

**Consequences of Compromising Principles (Tutorial 1, Q2):**

*   **Confidentiality Compromised:** Sensitive data leakage (personal info, trade secrets), identity theft, loss of reputation, legal penalties, blackmail.
*   **Authentication Compromised:** Unauthorized access, impersonation, privilege escalation, data breaches, system takeover.
*   **Integrity Compromised:** Corrupted data leading to incorrect decisions, financial loss (e.g., modified bank balance), reputational damage, loss of trust, system malfunction.
*   **Non-Repudiation Compromised:** Inability to prove actions, difficulty in resolving disputes, legal challenges, users can deny harmful actions.
*   **Access Control Compromised:** Unauthorized access to sensitive data or systems, privilege escalation, data breaches, system misuse.
*   **Availability Compromised:** Disruption of services, financial loss due to downtime, loss of productivity, reputational damage, inability to perform critical functions.

---

### **Topic 2: Asymmetric Encryption or Public Key Cryptography (in calculation)**

(Primarily from Chapter 3, Slides 63-64 for RSA, Slides 78-82 for Diffie-Hellman)

**I. RSA (Rivest-Shamir-Adleman) Algorithm**

*   **Basis:** Difficulty of factoring the product of two large prime numbers.
*   **Key Generation (Chapter 3, Slide 63):**
    1.  **Select two distinct large prime numbers, `p` and `q`.**
    2.  **Compute `n = p * q`** (modulus for both public and private keys).
    3.  **Compute Euler's totient function: `φ(n) = (p-1) * (q-1)`**.
    4.  **Choose an integer `e` (public exponent)** such that `1 < e < φ(n)` and `gcd(e, φ(n)) = 1` (i.e., `e` and `φ(n)` are coprime).
    5.  **Compute `d` (private exponent)** such that `d * e ≡ 1 (mod φ(n))`. This means `d` is the modular multiplicative inverse of `e` modulo `φ(n)`. (Can be found using the Extended Euclidean Algorithm).
    6.  **Public Key:** `KU = {e, n}`
    7.  **Private Key:** `KR = {d, n}`
    8.  `p`, `q`, and `φ(n)` must be kept secret.

*   **Encryption (Plaintext M, Ciphertext C):**
    `C = M^e mod n`

*   **Decryption (Ciphertext C, Plaintext M):**
    `M = C^d mod n`

*   **Example Calculation (Chapter 3, Slide 64, modified for clarity):**
    Let `p = 7`, `q = 19`.
    1.  `n = p * q = 7 * 19 = 133`.
    2.  `φ(n) = (p-1) * (q-1) = (7-1) * (19-1) = 6 * 18 = 108`.
    3.  Choose `e = 5`. Check: `1 < 5 < 108`. `gcd(5, 108) = 1` (since 108 is not divisible by 5). So, `e=5` is valid.
    4.  Find `d` such that `d * 5 ≡ 1 (mod 108)`.
        *   This means `5d = 1 + k*108` for some integer `k`.
        *   We need `(k*108 + 1)` to be divisible by 5.
        *   If `k=1`, `108+1 = 109` (not div by 5)
        *   If `k=2`, `216+1 = 217` (not div by 5)
        *   If `k=3`, `324+1 = 325`. `325 / 5 = 65`. So `d = 65`.
        *   (Slide calculation: `5d-1 = 108 * k` -> `5d - 1 = 108` (k=1) -> `5d = 109` (no int solution); `5d-1 = 108*2 = 216` (k=2) -> `5d = 217` (no int); `5d-1 = 108*3 = 324` (k=3) -> `5d = 325` -> `d=65`).
    5.  **Public Key:** `KU = {5, 133}`
    6.  **Private Key:** `KR = {65, 133}`

    Let Plaintext `M = 4` (from Exam Question format).
    *   **Encryption:** `C = M^e mod n = 4^5 mod 133`
        `4^1 = 4`
        `4^2 = 16`
        `4^3 = 64`
        `4^4 = 256 ≡ 123 (mod 133)`  (since 256 = 1*133 + 123)
        `4^5 = 4^4 * 4^1 ≡ 123 * 4 (mod 133)`
        `123 * 4 = 492`
        `492 mod 133`:  `492 = 3 * 133 + 93` (since 3*133 = 399; 492-399 = 93)
        So, `C = 93`.

    *   **Decryption:** `M = C^d mod n = 93^65 mod 133`
        (This requires modular exponentiation, often by repeated squaring, which is too complex for quick exam calculation without tools, unless numbers are very small or a calculator with mod function is allowed. The exam question usually provides simpler numbers or asks for the steps/formulae rather than full large power calculation.)
        The question on slide 64 is an *illustration of key generation*, not encryption/decryption of a message.

**II. Diffie-Hellman Key Exchange**

*   **Purpose:** Allows two parties (Alice and Bob) to establish a shared secret key over an insecure public channel, which can then be used for symmetric encryption.
*   **Global Public Elements (known to everyone, including attackers):**
    *   `q`: A large prime number.
    *   `α`: A primitive root modulo `q`. (This means powers of `α` mod `q` generate all numbers from 1 to `q-1`).

*   **Key Exchange Steps (Chapter 3, Slides 79-82):**
    1.  **Alice:**
        *   Selects a private key (a random integer) `XA` such that `XA < q`.
        *   Calculates her public key `YA = α^XA mod q`.
        *   Sends `YA` to Bob.
    2.  **Bob:**
        *   Selects a private key (a random integer) `XB` such that `XB < q`.
        *   Calculates his public key `YB = α^XB mod q`.
        *   Sends `YB` to Alice.
    3.  **Alice (computes shared secret key K):**
        `K = YB^XA mod q = (α^XB)^XA mod q = α^(XB*XA) mod q`
    4.  **Bob (computes shared secret key K):**
        `K = YA^XB mod q = (α^XA)^XB mod q = α^(XA*XB) mod q`
    *   Both Alice and Bob arrive at the same shared secret key `K`.

*   **Example Calculation (Chapter 3, Slide 81-82):**
    *   Global Public Elements: `q = 23`, `α = 5`.
    *   **Alice:**
        *   Chooses private key `XA = 4`.
        *   Calculates public key `YA = α^XA mod q = 5^4 mod 23`.
            `5^1 = 5`
            `5^2 = 25 ≡ 2 (mod 23)`
            `5^3 = 5^2 * 5 ≡ 2 * 5 = 10 (mod 23)`
            `5^4 = 5^3 * 5 ≡ 10 * 5 = 50 ≡ 4 (mod 23)` (since 50 = 2*23 + 4)
            So, `YA = 4`. Alice sends `4` to Bob.
    *   **Bob:**
        *   Chooses private key `XB = 3`.
        *   Calculates public key `YB = α^XB mod q = 5^3 mod 23`.
            `5^3 = 125`.
            `125 mod 23`: `125 = 5 * 23 + 10` (since 5*23 = 115; 125-115 = 10)
            So, `YB = 10`. Bob sends `10` to Alice.
    *   **Alice (computes shared secret K):**
        `K = YB^XA mod q = 10^4 mod 23`.
        `10^1 = 10`
        `10^2 = 100 ≡ 8 (mod 23)` (since 100 = 4*23 + 8; 4*23 = 92)
        `10^3 = 10^2 * 10 ≡ 8 * 10 = 80 ≡ 11 (mod 23)` (since 80 = 3*23 + 11; 3*23 = 69)
        `10^4 = 10^3 * 10 ≡ 11 * 10 = 110 ≡ 18 (mod 23)` (since 110 = 4*23 + 18; 4*23 = 92)
        So, Alice computes `K = 18`.
    *   **Bob (computes shared secret K):**
        `K = YA^XB mod q = 4^3 mod 23`.
        `4^1 = 4`
        `4^2 = 16`
        `4^3 = 64`.
        `64 mod 23`: `64 = 2 * 23 + 18` (since 2*23 = 46; 64-46 = 18)
        So, Bob computes `K = 18`.
    *   Both arrive at the shared secret key `K = 18`.

---

## Part 2: Flashcards

---

(All questions first, then all answers)

**Flashcard Questions: Security Principles Usage**

66. Q: What is the primary goal of Confidentiality in computer security?
67. Q: Give an example of how Integrity can be compromised in a university system.
68. Q: How does a DoS attack primarily affect a security principle? Which principle?
69. Q: Why is strong Authentication crucial before granting access to resources?
70. Q: How do digital signatures support Non-Repudiation?
71. Q: If an attacker bypasses physical locks to a server room, which broad security principle is immediately at risk regarding the data on those servers?

**Flashcard Questions: Asymmetric Encryption Calculations**

72. Q: In RSA, what do `p` and `q` represent, and what is their key characteristic?
73. Q: How is `n` calculated in RSA?
74. Q: What is `φ(n)` (Euler's totient function) in RSA, and how is it calculated if `p` and `q` are known?
75. Q: What conditions must the public exponent `e` satisfy in RSA?
76. Q: What is the mathematical relationship between the public exponent `e`, private exponent `d`, and `φ(n)` in RSA?
77. Q: If Alice's RSA public key is `{e, n}` and she receives ciphertext `C`, how does she decrypt it to get plaintext `M` (formula)?
78. Q: In Diffie-Hellman, what are `q` and `α`? Are they secret?
79. Q: In Diffie-Hellman, if Alice's private key is `XA` and Bob's public key is `YB`, what formula does Alice use to compute the shared secret key `K`?
80. Q: Can an attacker listening on the public channel derive the Diffie-Hellman shared secret `K` if they know `q`, `α`, Alice's public key `YA`, and Bob's public key `YB`? Why or why not?

---

**Flashcard Answers**

*Security Principles Usage Answers:*

66. A: To ensure that information is not disclosed to unauthorized individuals, entities, or processes.
67. A: An unauthorized person altering a student's grades in the university database.
68. A: It primarily affects Availability, making the system or service inaccessible to legitimate users.
69. A: To ensure that only verified, legitimate users are granted permissions, preventing unauthorized access and potential misuse of resources.
70. A: By cryptographically linking a message (via its hash) to the sender's private key, it provides verifiable proof that the specific sender originated the message and cannot later deny it.
71. A: Confidentiality (attacker can view data), Integrity (attacker can alter data), and Availability (attacker can damage/disable servers). *(Confidentiality is often the most immediate concern for data theft)*

*Asymmetric Encryption Calculations Answers:*

72. A: `p` and `q` are two distinct large prime numbers.
73. A: `n = p * q`.
74. A: `φ(n)` is Euler's totient function, representing the count of positive integers less than or equal to `n` that are relatively prime to `n`. For RSA, `φ(n) = (p-1) * (q-1)`.
75. A: `1 < e < φ(n)` and `gcd(e, φ(n)) = 1` (e must be coprime to `φ(n)`).
76. A: `d * e ≡ 1 (mod φ(n))`, meaning `d` is the modular multiplicative inverse of `e` modulo `φ(n)`.
77. A: `M = C^d mod n`.
78. A: `q` is a large prime number, and `α` is a primitive root modulo `q`. They are global public elements, known to everyone and not secret.
79. A: `K = YB^XA mod q`.
80. A: No (theoretically, for large enough numbers). Because to find `K`, the attacker would need to know either Alice's private key `XA` or Bob's private key `XB`. Deriving `XA` from `YA = α^XA mod q` (or `XB` from `YB`) is the discrete logarithm problem, which is computationally infeasible for large `q`.

---

## Part 3: Mini-Tests

---

**Mini-Test 1: Security Principles Application (10 minutes)**

1.  A bank customer discovers that their account balance has been incorrectly reduced. An investigation reveals an unauthorized transaction was posted. Which core security principle was primarily violated? Explain your reasoning. (3 marks)
2.  A company's website becomes inaccessible to all users due to a massive flood of traffic from multiple sources. Which security principle is under attack? Suggest one measure to improve this. (3 marks)
3.  To ensure an important contract sent electronically is genuinely from the CEO and hasn't been altered, what two security principles are most critical to uphold, and what cryptographic mechanism typically addresses both? (4 marks)

**Mini-Test 1: Answers**

1.  **Integrity** was primarily violated. The accuracy and trustworthiness of the account balance data were compromised by an unauthorized modification. (Confidentiality might also be a concern if the attacker viewed the balance, and Authentication if the attacker impersonated someone, but the direct result described is an integrity failure).
2.  **Availability** is under attack (likely a Distributed Denial of Service - DDoS). Measure: Implement DDoS mitigation services, use firewalls with traffic filtering, or increase server/network capacity.
3.  The two principles are **Authentication** (of the CEO as the sender) and **Integrity** (of the contract content). The cryptographic mechanism is a **Digital Signature**.

---

**Mini-Test 2: Asymmetric Cryptography Calculations (20 minutes)**

1.  **RSA Key Generation:**
    Given prime numbers `p = 3` and `q = 11`.
    The public exponent `e` is chosen as `e = 7`.
    (a) Calculate `n`. (1 mark)
    (b) Calculate `φ(n)`. (1 mark)
    (c) Find the private exponent `d`. Show your steps. (3 marks)

2.  **RSA Encryption:**
    Using the public key `KU = {7, 33}` (derived from p=3, q=11, e=7).
    Encrypt the plaintext message `M = 2`. Show your calculation. (2 marks)

3.  **Diffie-Hellman Key Exchange:**
    Alice and Bob use `q = 13` and `α = 2`.
    Alice chooses private key `XA = 5`. Bob chooses private key `XB = 4`.
    (a) Calculate Alice's public key `YA`. (2 marks)
    (b) Calculate Bob's public key `YB`. (2 marks)
    (c) Calculate the shared secret key `K` from Alice's perspective. (2 marks)
    (d) Calculate the shared secret key `K` from Bob's perspective. (2 marks)

**Mini-Test 2: Answers**

1.  **RSA Key Generation:** `p = 3`, `q = 11`, `e = 7`.
    (a) `n = p * q = 3 * 11 = 33`.
    (b) `φ(n) = (p-1) * (q-1) = (3-1) * (11-1) = 2 * 10 = 20`.
    (c) Find `d` such that `d * 7 ≡ 1 (mod 20)`.
        `7d = 1 + k*20`.
        If `k=1`, `7d = 21` => `d = 3`.
        Check: `(3 * 7) mod 20 = 21 mod 20 = 1`. So, `d = 3`.

2.  **RSA Encryption:** `KU = {7, 33}`, `M = 2`.
    `C = M^e mod n = 2^7 mod 33`.
    `2^1 = 2`
    `2^2 = 4`
    `2^3 = 8`
    `2^4 = 16`
    `2^5 = 32`
    `2^6 = 64 ≡ 31 (mod 33)` or `2^6 = 2 * 32 ≡ 2 * (-1) = -2 ≡ 31 (mod 33)`
    `2^7 = 2 * 31 = 62 ≡ 29 (mod 33)`.
    So, `C = 29`.

3.  **Diffie-Hellman Key Exchange:** `q = 13`, `α = 2`, `XA = 5`, `XB = 4`.
    (a) Alice's public key `YA = α^XA mod q = 2^5 mod 13`.
        `2^1 = 2`
        `2^2 = 4`
        `2^3 = 8`
        `2^4 = 16 ≡ 3 (mod 13)`
        `2^5 = 2 * 3 = 6 (mod 13)`.
        So, `YA = 6`.
    (b) Bob's public key `YB = α^XB mod q = 2^4 mod 13`.
        From above, `2^4 mod 13 = 3`.
        So, `YB = 3`.
    (c) Shared secret `K` (Alice's perspective) = `YB^XA mod q = 3^5 mod 13`.
        `3^1 = 3`
        `3^2 = 9`
        `3^3 = 27 ≡ 1 (mod 13)`
        `3^4 = 3 * 1 = 3 (mod 13)`
        `3^5 = 3 * 3 = 9 (mod 13)`.
        So, `K = 9`.
    (d) Shared secret `K` (Bob's perspective) = `YA^XB mod q = 6^4 mod 13`.
        `6^1 = 6`
        `6^2 = 36 ≡ 10 (mod 13)` (since 36 = 2*13 + 10)
        `6^3 = 6 * 10 = 60 ≡ 8 (mod 13)` (since 60 = 4*13 + 8)
        `6^4 = 6 * 8 = 48 ≡ 9 (mod 13)` (since 48 = 3*13 + 9)
        So, `K = 9`.

---

## Part 4: Mock Exam Questions

---

**Question 1 (Security Principles & Asymmetric Crypto Concepts - 25 marks)**

**(a)** For each of the following scenarios, identify the primary security principle that has been compromised. Justify your answer for each.
    **(i)** An attacker successfully intercepts and reads an email containing confidential merger plans between two companies. (3 marks)
    **(ii) A disgruntled ex-employee logs into the company's sales database and changes all sales figures for the last quarter to zero. (3 marks)
    **(iii)** During a major online sales event, a retailer's website crashes due to an overwhelming number of fake requests, preventing legitimate customers from making purchases. (3 marks)

**(b)** Describe the purpose of using a public key and a private key in asymmetric cryptography for:
    **(i)** Ensuring message confidentiality. (3 marks)
    **(ii)** Creating a digital signature. (3 marks)

**(c)** In the Diffie-Hellman key exchange, Alice and Bob publicly agree on a prime `q` and a primitive root `α`. Alice chooses a secret `XA` and Bob chooses a secret `XB`.
    **(i)** What does Alice compute and send to Bob? (Write the formula) (2 marks)
    **(ii)** What does Bob compute and send to Alice? (Write the formula) (2 marks)
    **(iii) How does Alice compute the final shared secret key K after receiving Bob's transmission? (Write the formula) (2 marks)
    **(iv)** Why is this key exchange secure even if an attacker intercepts the public transmissions of Alice and Bob? (4 marks)

**Question 2 (RSA Calculation - 25 marks)**

Alice wishes to use the RSA algorithm to receive encrypted messages. She selects the prime numbers `p = 5` and `q = 13`. She chooses her public exponent `e = 7`.

**(a)** Calculate the value of `n` for Alice's RSA key pair. (3 marks)
**(b)** Calculate the value of `φ(n)` (Euler's totient function) for Alice's setup. (4 marks)
**(c)** Determine Alice's private exponent `d`. Show all steps in your calculation. (8 marks)
**(d)** Bob wants to send the plaintext message `M = 2` to Alice. Encrypt this message using Alice's public key. Show your calculation. (6 marks)
**(e)** If Alice receives a ciphertext `C = 3`, show the formula she would use to decrypt it, using the variables `C`, `d`, and `n`. (You do not need to calculate the final plaintext for this part). (4 marks)

---

**Mock Exam Model Answers (Key Points)**

**Question 1:**

**(a)**
    **(i) Confidentiality.** The merger plans, which were intended to be secret, were disclosed to an unauthorized party.
    **(ii) Integrity.** The accuracy and trustworthiness of the sales figures were compromised by unauthorized modification.
    **(iii) Availability.** The website was made inaccessible to legitimate users, preventing them from using the service.

**(b)**
    **(i) Message Confidentiality:** The sender encrypts the message using the *recipient's public key*. Only the recipient, who possesses the corresponding *private key*, can decrypt and read the message.
    **(ii) Creating a Digital Signature:** The sender encrypts a hash of the message using their *own private key*. This signature can be verified by anyone using the sender's *public key*, proving authenticity and integrity.

**(c)**
    **(i) Alice computes and sends:** `YA = α^XA mod q`
    **(ii) Bob computes and sends:** `YB = α^XB mod q`
    **(iii) Alice computes K:** `K = YB^XA mod q`
    **(iv) Security:** An attacker intercepts `YA` and `YB` (and knows `q`, `α`). To find the shared secret `K`, the attacker would need to find `XA` from `YA` (or `XB` from `YB`). This is the discrete logarithm problem (`XA = log_α YA mod q`), which is computationally infeasible for large prime `q`.

**Question 2:**

`p = 5`, `q = 13`, `e = 7`.

**(a) Calculate `n`:**
    `n = p * q = 5 * 13 = 65`. (3 marks)

**(b) Calculate `φ(n)`:**
    `φ(n) = (p-1) * (q-1) = (5-1) * (13-1) = 4 * 12 = 48`. (4 marks)

**(c) Determine `d`:**
    We need `d * e ≡ 1 (mod φ(n))`, so `d * 7 ≡ 1 (mod 48)`.
    This means `7d = 1 + k*48` for some integer `k`.
    We need `(1 + k*48)` to be divisible by 7.
    If `k=1`: `1 + 48 = 49`. `49 / 7 = 7`.
    So, `d = 7`.
    Check: `(7 * 7) mod 48 = 49 mod 48 = 1`.
    Therefore, Alice's private exponent `d = 7`. (8 marks)

**(d) Encrypt `M = 2`:**
    Alice's public key is `KU = {e, n} = {7, 65}`.
    `C = M^e mod n = 2^7 mod 65`.
    `2^1 = 2`
    `2^2 = 4`
    `2^3 = 8`
    `2^4 = 16`
    `2^5 = 32`
    `2^6 = 64 ≡ -1 (mod 65)`
    `2^7 = 2 * 64 = 128`.
    `128 mod 65`: `128 = 1 * 65 + 63`.
    So, `C = 63`. (6 marks)

**(e) Decryption formula for `C = 3`:**
    Alice uses her private key `KR = {d, n} = {7, 65}`.
    The formula to decrypt ciphertext `C` is:
    `M = C^d mod n`
    So, for `C = 3`, `M = 3^7 mod 65`. (4 marks)

---

This set of notes, flashcards, mini-tests, and mock exam questions should provide a targeted revision for the specified scope. Good luck with your finals!



Okay, let's create targeted revision materials for **Means of Authentication** and **Security Policy** based on your BAIT1093 course slides.

---

## Part 1: Revision Notes

---

### **Topic 1: Means of Authentication**

(Primarily from Chapter 4, Slides 5-70)

**I. Overview (Chapter 4, Slide 5)**

Authentication verifies a user's identity based on one or more factors. The four primary means are:

1.  **Something the individual knows:** (e.g., Password, PIN, answers to prearranged questions)
2.  **Something the individual possesses (token):** (e.g., Smartcard, electronic keycard, physical key, security token)
3.  **Something the individual is (static biometrics):** (e.g., Fingerprint, retina, face)
4.  **Something the individual does (dynamic/behavioral biometrics):** (e.g., Voice pattern, handwriting, typing rhythm)

**II. Password Authentication (Chapter 4, Slides 6-29)**

*   **Definition:** Secret combination of letters, numbers, and/or characters.
*   **Challenges:**
    *   Difficult for users to memorize complex/long/unique passwords for multiple accounts.
    *   Password expiration policies requiring frequent memorization.
*   **User Shortcuts leading to Weak Passwords:**
    *   Using common words, short words, predictable sequences (abc123), personal info.
    *   Reusing passwords.
    *   Predictable patterns in stronger passwords:
        *   *Appending:* `simon1`, `food$6`
        *   *Replacing:* `passw0rd`, `s1mon`, `$imon`
*   **Attacks on Passwords:**
    *   **Pass the Hash Attack / Password Cracker (Slides 13-15):**
        *   Passwords are stored as hashes (digests), not plaintext.
        *   Attackers steal the hash file.
        *   *Pass the Hash:* Use the stolen hash directly to impersonate (e.g., NTLM vulnerability).
        *   *Password Cracker:* Create candidate digests (from wordlists, brute-force) and compare against stolen digests to find the original password.
    *   **Password Spraying (Slide 16):** Trying a few common passwords against many user accounts to avoid account lockout.
    *   **Brute Force Attack (Slides 17-18):** Trying every possible combination of characters.
        *   *Online:* Continuously attacking the same account with different passwords.
        *   *Offline:* Uses a stolen digest file and password cracking software to generate and match candidates. Slowest but most thorough.
    *   **Rule Attack (Slides 19-22):** Conducts statistical analysis on known (often stolen) passwords to create masks (e.g., `?u?l?l?l?d?d?d?d` for Uppercase + 4 lowercase + 4 digits) to guide the cracking process and reduce time.
    *   **Dictionary Attack (Slides 23-24):** Creates digests of common dictionary words and compares them against stolen digests.
    *   **Rainbow Tables (Slides 25-26):** Large pre-generated data sets of candidate digests (chains). Faster than dictionary attacks and require less memory on the attacking machine.
    *   **Password Collections (Slides 27-28):** Using lists of previously leaked passwords as candidates for attacks.
    *   **Combined Attack Tools (Slide 29):** Attackers often use a sequence of tools/methods.
*   **Password Security Solutions (Chapter 4, Slides 75-79):**
    *   **Protecting Password Digests:**
        *   **Salts:** Random strings added to plaintext passwords *before* hashing. Ensures unique digests even for identical passwords, prevents dictionary/rainbow table attacks on unsalted hashes.
        *   **Key Stretching:** Slows down password hashing (e.g., bcrypt, PBKDF2, Argon2) by increasing computation time, making brute-force attacks harder.
    *   **Managing Passwords:**
        *   **Length over complexity:** Longer passwords are generally harder to crack.
        *   **Password Vaults:** Secure repositories for storing passwords (generators, online vaults, management apps).
        *   **Password Keys (Hardware):** Hardware-based solutions, often more secure than software vaults.

**III. Token-Based Authentication (Something You Have) (Chapter 4, Slides 30-48)**

*   Often used with passwords for **Multifactor Authentication (MFA)** or **Two-Factor Authentication (2FA)**.
*   **Types of Tokens:**
    *   **Specialized Devices:**
        *   **Smart Cards (Slides 32-33):** Credit-card sized, hold info. Contact or contactless.
            *   *Disadvantages:* Requires readers, magnetic stripe cards vulnerable to cloning/skimming.
        *   **Windowed Tokens (OTP) (Slides 34-39):** Display a dynamic One-Time Password.
            *   **TOTP (Time-based OTP):** Changes after a set time (e.g., 30-60s). Token and server share algorithm & time.
            *   **HOTP (HMAC-based OTP):** Event-driven, changes on specific event (e.g., PIN entry on token).
            *   *Disadvantage:* Cumbersome (manual, time-sensitive entry).
    *   **Smartphones (Slides 40-44):**
        *   *Phone call verification:* Automated call for approval.
        *   *SMS text message (OTP):* Receive OTP via SMS. *Insecure: Phishable, SMS can be intercepted.*
        *   *Authentication App:* Push notification for approval. *Insecure if phone is compromised by malware.*
    *   **Security Keys (Hardware Tokens) (Slides 45-48):**
        *   Dongle (USB/Lightning/NFC). Contains cryptographic info.
        *   **Attestation:** Key pair "burned" in during manufacturing, specific to device model. Cryptographically proves device genuineness when registering new credentials.
        *   More secure than OTPs (not easily intercepted/phished). Recommended alternative.

**IV. Biometric Authentication (Something You Are/Do) (Chapter 4, Slides 49-68)**

*   Uses unique features and characteristics of an individual.
*   **Physiological Biometrics (Slides 50-63):** Based on body part functions.
    *   *Specialized Scanners:* Retinal (unique capillary patterns), Fingerprint (ridges/valleys; static or dynamic scanners), Vein (palm/finger patterns), Gait recognition (manner of walking).
    *   *Standard Input Devices:* Voice Recognition (unique voice characteristics via microphone), Iris Scanner (unique iris patterns via webcam), Facial Recognition (distinguishable "landmarks" or nodal points via webcam).
    *   *Disadvantages:* Cost of specialized scanners, not foolproof (false rejections/acceptances), can be "tricked" (e.g., lifted fingerprints), user privacy concerns, efficacy rates.
*   **Cognitive Biometrics (Slides 64-68):** Based on perception, thought process, understanding (knowledge-based).
    *   *Windows Picture Password:* User selects a picture and performs specific gestures (tap, line, circle) on points of interest.
    *   *Memorable Events:* Recalling details about personal events/experiences.
    *   *Vulnerability:* Users may choose predictable patterns or easily guessable information.

**V. Behavioral Biometrics Authentication (Chapter 4, Slides 69-70)**

*   Authenticates based on unique actions performed by the user.
*   **Keystroke Dynamics:** Recognizes unique typing rhythm.
    *   Measures *dwell time* (key press duration) and *flight time* (time between keystrokes).
    *   Multiple samples create a user template for comparison.
    *   Convenient, no specialized hardware.

---

### **Topic 2: Security Policy**

(Primarily from Chapter 7, Slides 9-21; Chapter 2, Slides 61-66 for malware context)

**I. Information Security Policy (Chapter 7, Slides 9-10)**

*   **Definition:** An important element of an effective security strategy. A set of written practices and procedures that all employees must follow to ensure the confidentiality, integrity, and availability (CIA) of data and resources.
*   **Purpose:**
    *   Defines business expectations for security.
    *   Outlines how security objectives are to be achieved.
    *   Describes consequences for failure to comply.
    *   Aims to protect the organization.
*   **Structure:** Many organizations opt for specific, targeted policies rather than one large policy for easier user digestion.

**II. Examples of Information Security Policies (Chapter 7, Slides 10-20; Chapter 2, Slides 62-66)**

*   **Network Security Policy (Ch7, Slide 11):** General rules for network access, architecture, security environments, and enforcement.
*   **Workstation Policy (Ch7, Slide 12):** General security for workstations (antivirus, locking unattended, password usage, patching).
*   **Acceptable Use Policy (AUP) (Ch7, Slide 12):** Defines acceptable/unacceptable use of internet, email, social networking, file transfer.
*   **Clean Desk Policy (Ch7, Slide 13):** Guidelines for maintaining a clean, uncluttered desk to protect sensitive information from being exposed.
*   **Remote Access Policy (Ch7, Slide 13):** Defines remote access, who is permitted, permitted devices/OS, and methods (VPNs).
*   **Password Policy (Ch7, Slide 14):** Standards for strong password creation, protection, and frequency of change.
*   **Account Management Policy (Ch7, Slide 14):** Standards for creation, administration, use, and removal of user accounts.
*   **Email Security Policy (Ch7, Slide 15):** Rules for using company email (sending, receiving, storing).
*   **Log Management Policy (Ch7, Slide 15):** Guidelines for managing logs to enhance security, performance, resource management, and compliance.
*   **Security Incident Management Policy (Ch7, Slide 16):** Requirements for reporting and responding to security incidents.
*   **Personal Device Acceptable Use and Security (BYOD) Policy (Ch7, Slide 17):** Standards for end-users accessing corporate data using personal devices.
*   **Patch Management Policy (Ch7, Slide 18):** Procedures for applying software "patches" to mitigate vulnerabilities.
*   **Server Security Policy (Ch7, Slide 19):** Standards for base configuration of internal server equipment.
*   **Systems Monitoring and Auditing Policy (Ch7, Slide 20):** Defines use of system monitoring (real-time) and auditing (after-the-fact) to detect inappropriate actions.
*   **Malware-Specific Policies (from Chapter 2 Countermeasures):**
    *   **Social Engineering Awareness Policy (Ch2, Slide 63):** Guidelines to provide awareness and define procedures for social engineering threats.
    *   **Server Malware Protection Policy (Ch2, Slide 64):** Outlines requirements for anti-virus/anti-spyware on servers.
    *   **Software Installation Policy (Ch2, Slide 65):** Governs software installation on company devices to minimize risks.
    *   **Removable Media Policy (Ch2, Slide 66):** Minimizes risk of loss/exposure of sensitive info and malware infections via removable media.

**III. Importance and Role of Security Policies**

*   **Foundation for Security Program:** Provide a baseline and framework for all security efforts.
*   **Clarify Expectations:** Inform employees of their responsibilities regarding security.
*   **Consistency:** Ensure security measures are applied consistently across the organization.
*   **Legal and Regulatory Compliance:** Help meet legal, contractual, and regulatory obligations.
*   **Due Diligence/Care:** Demonstrates the organization is taking steps to protect assets.
*   **Incident Response:** Provide procedures for handling security incidents.
*   **Risk Management:** Help identify, assess, and mitigate risks.
*   A template for a comprehensive IT Security Policy can be found (Chapter 7, Slide 21).

---

## Part 2: Flashcards

---

(All questions first, then all answers)

**Flashcard Questions: Means of Authentication**

81. Q: What are the four fundamental means (categories) of authenticating a user's identity?
82. Q: What is a primary challenge associated with password authentication for users?
83. Q: Describe a "Pass the Hash" attack.
84. Q: How does a "Rule Attack" differ from a standard "Brute Force Attack" on passwords?
85. Q: What is the purpose of "salting" passwords before hashing them?
86. Q: What is "key stretching" in the context of password security?
87. Q: What is Two-Factor Authentication (2FA)?
88. Q: What is a TOTP and how does it typically work?
89. Q: What is a major security concern with using SMS text messages for OTP delivery?
90. Q: What is "attestation" in relation to hardware security keys?
91. Q: Name two examples of physiological biometrics and one example of cognitive biometrics.
92. Q: What is keystroke dynamics based on?

**Flashcard Questions: Security Policy**

93. Q: What is an Information Security Policy?
94. Q: State two main purposes of having security policies in an organization.
95. Q: What does an Acceptable Use Policy (AUP) typically define?
96. Q: What is the main goal of a Patch Management Policy?
97. Q: How does a Removable Media Policy help in preventing malware attacks?
98. Q: What kind of information would an Email Security Policy cover?
99. Q: Why is a Security Incident Management Policy important?
100.Q: A "Clean Desk Policy" primarily aims to protect against what type of information exposure?

---

**Flashcard Answers**

*Means of Authentication Answers:*

81. A: Something you know, something you have, something you are, and something you do.
82. A: Difficulty in memorizing and managing multiple complex, unique passwords.
83. A: An attack where an attacker obtains a user's password hash and uses it to authenticate to a system or service without needing the plaintext password.
84. A: A Rule Attack uses statistical analysis of known passwords to create masks (patterns) to guide the cracking process, making it more efficient than a Brute Force Attack which tries all possible combinations.
85. A: To ensure that identical passwords produce different hash values, rendering pre-computed tables (like rainbow tables) ineffective and increasing the difficulty of cracking multiple accounts if they share the same password.
86. A: A technique that makes password hashing computationally more intensive (slower) by using specialized algorithms or multiple iterations, thus making brute-force or dictionary attacks much more time-consuming.
87. A: An authentication method requiring a user to provide two distinct forms of identification from different categories (e.g., something you know + something you have).
88. A: Time-based One-Time Password. It's an OTP that is valid only for a short period (e.g., 30-60 seconds), generated by an algorithm shared between a token/app and an authentication server, synchronized by time.
89. A: SMS messages can be intercepted (e.g., via SIM swapping, malware on phone) or OTPs can be phished from the user.
90. A: A feature where the security key can cryptographically prove its genuineness (e.g., its manufacturer and model) to a service, often using a key pair embedded during manufacturing.
91. A: Physiological: Fingerprint, Facial Recognition. Cognitive: Picture Password (gestures on an image).
92. A: The unique typing rhythm of a user, including dwell time (how long a key is pressed) and flight time (the time between key presses).

*Security Policy Answers:*

93. A: A set of written practices, procedures, and guidelines that direct employee behavior to ensure the confidentiality, integrity, and availability of an organization's data and resources.
94. A: 1. To define acceptable and unacceptable behavior regarding information systems. 2. To provide a framework for security measures and incident response, protecting organizational assets.
95. A: It defines the permissible and prohibited uses of an organization's IT resources, such as internet access, email systems, and network services.
96. A: To ensure that software vulnerabilities are identified and remediated in a timely manner by applying security patches.
97. A: By setting rules for the use of USB drives and other portable storage, it helps prevent the introduction of malware from infected external devices and unauthorized data exfiltration.
98. A: Rules for sending, receiving, storing company email, handling attachments, and protecting sensitive information communicated via email.
99. A: It provides a structured approach for an organization to identify, respond to, manage, and learn from security incidents, minimizing damage and preventing recurrence.
100.A: It aims to protect sensitive information (e.g., on papers, sticky notes, unlocked screens) from being casually viewed or stolen by unauthorized individuals in the physical workspace.

---

## Part 3: Mini-Tests

---

**Mini-Test 1: Means of Authentication (15 minutes)**

1.  **Multiple Choice (Choose one):** Which password attack method involves using a large pre-generated dataset of hash values and their corresponding plaintexts, organized in chains?
    a)  Dictionary Attack
    b)  Brute Force Attack
    c)  Rainbow Table Attack
    d)  Password Spraying

2.  **Short Answer:** Explain the primary difference between a TOTP (Time-based One-Time Password) and an HOTP (HMAC-based One-Time Password).

3.  **Scenario:** A user logs into their work computer using a password and then plugs in a hardware security key which they tap to complete the login. What category of authentication factor does the security key represent? Is this an example of MFA? Explain.

4.  **List:** Name two common vulnerabilities or disadvantages associated with using physiological biometrics for authentication.

**Mini-Test 1: Answers**

1.  c) Rainbow Table Attack
2.  A TOTP changes after a fixed period of time (e.g., every 30 or 60 seconds), synchronized by time between the token and server. An HOTP is event-driven and changes when a specific event occurs (e.g., a button press on the token or a counter increment).
3.  The hardware security key represents "something you have." Yes, this is an example of MFA (specifically 2FA) because it combines "something you know" (the password) with "something you have" (the security key).
4.  Two disadvantages:
    *   Can be "tricked" or spoofed (e.g., lifted fingerprints, high-resolution photos for facial recognition).
    *   Not foolproof: Can have false acceptance rates (FAR) or false rejection rates (FRR).
    *   Cost of specialized biometric scanners (for some types like retinal/vein).
    *   User privacy concerns about storing biometric data.

---

**Mini-Test 2: Security Policy (15 minutes)**

1.  **True/False (Justify your answer):** An Information Security Policy is primarily a technical document intended only for IT staff.

2.  **Short Answer:** What is the main objective of an "Acceptable Use Policy" (AUP)?

3.  **Scenario:** An employee frequently downloads and installs unapproved software from the internet onto their company laptop for personal use. Which specific type of security policy would directly address and aim to prevent this behavior? Briefly explain how.

4.  **List:** Name three distinct types of security policies (other than an AUP) that an organization might implement.

**Mini-Test 2: Answers**

1.  False. While it has technical implications, an Information Security Policy is a high-level document that applies to *all* employees and users of an organization's IT systems. It outlines expected behaviors, responsibilities, and consequences related to information security.
2.  The main objective of an AUP is to define for users what is considered permissible and prohibited use of the organization's information systems, network resources, and internet access, thereby reducing risks and protecting assets.
3.  A "Software Installation Policy" would directly address this. It would define the rules and procedures for installing software on company devices, likely prohibiting unauthorized downloads or requiring approval for all installations, thus preventing the introduction of potentially malicious or unlicensed software.
4.  Password Policy, Remote Access Policy, Email Security Policy (or any other valid distinct examples from the notes).

---

## Part 4: Mock Exam Questions

---

**Question 1 (Means of Authentication - 25 marks)**

**(a)** Compare and contrast "Password Spraying" and "Offline Brute Force Attack" as methods of password cracking. Include at least one similarity and two distinct differences. (6 marks)

**(b)** Your company is considering implementing multifactor authentication (MFA) for all employees.
    **(i)** Explain what MFA is and why it is generally more secure than single-factor authentication. (4 marks)
    **(ii)** An employee suggests using SMS-based OTPs as the second factor due to its ease of use. Discuss two significant security concerns associated with this method. (6 marks)
    **(iii)** Propose a more secure alternative to SMS-OTP for the second factor and justify your choice by highlighting one key security advantage. (4 marks)

**(c)** Describe "keystroke dynamics" as a behavioral biometric authentication method. What two specific measurements does it typically use? (5 marks)

**Question 2 (Security Policy - 25 marks)**

**(a)** Define an "Information Security Policy" and explain its overall importance to an organization. (5 marks)

**(b)** For each of the following scenarios, identify the MOST relevant specific security policy (from your course material) that would help mitigate the risk, and briefly explain how that policy would apply:
    **(i)** An employee consistently leaves printouts containing customer financial data unattended on their desk overnight. (4 marks)
    **(ii) A remote employee connects to the company network using an unsecured public Wi-Fi, potentially exposing company data. (4 marks)
    **(iii)** A new malware variant is spreading rapidly by tricking users into clicking malicious links in emails. (4 marks)

**(c)** An organization has recently experienced several incidents where employees installed unauthorized peer-to-peer file-sharing software, leading to copyright infringement notices and potential malware infections.
    **(i)** Which specific security policy should be primarily reviewed and enforced to address this? (2 marks)
    **(ii)** List three key elements or rules that this policy should contain to effectively manage this risk. (6 marks)

---

**Mock Exam Model Answers (Key Points)**

**Question 1 (Means of Authentication):**

**(a) Password Spraying vs. Offline Brute Force:**
    *   **Similarity:** Both are methods aimed at discovering valid user passwords.
    *   **Differences:**
        1.  **Targeting:** Password Spraying uses a few common passwords against *many different accounts*. Offline Brute Force typically targets passwords for specific accounts from a *stolen hash file* or attempts all combinations against one account (online version, though "offline" is specified).
        2.  **Method/Resource:** Password Spraying is usually an online attack directly against login portals. Offline Brute Force (as named) works on a stolen hash file locally, without interacting with the live system for each guess, trying every possible combination or using dictionary/rule-based attacks on the hashes. Spraying tries to avoid lockout; brute force doesn't care if offline.

**(b) MFA:**
    **(i) Definition & Security:** MFA requires users to provide two or more distinct verification factors (from categories like something you know, have, or are) to gain access. It's more secure because even if one factor (e.g., password) is compromised, the attacker still needs to bypass the additional factor(s), significantly increasing the difficulty of unauthorized access.
    **(ii) SMS-OTP Concerns:**
        1.  **Interception:** SMS messages are not end-to-end encrypted and can be intercepted through various means (e.g., SS7 attacks, malicious apps on the phone, SIM swapping).
        2.  **Phishing:** Users can be tricked (phished) into revealing the OTP received via SMS to an attacker who has already obtained their password.
    **(iii) Secure Alternative & Advantage:**
        *   **Alternative:** Hardware Security Key (e.g., YubiKey) or an Authenticator App generating TOTPs (e.g., Google Authenticator).
        *   **Advantage of Hardware Key:** It is resistant to phishing (the key authenticates directly with the service, not via user input of a code) and remote attacks (requires physical presence). For Authenticator App: Not vulnerable to SMS interception, codes are generated locally.

**(c) Keystroke Dynamics:**
    *   A behavioral biometric authentication method that authenticates a user based on their unique typing rhythm and patterns when entering a known string (like a password or passphrase).
    *   It typically measures:
        1.  **Dwell time:** The duration a key is pressed.
        2.  **Flight time:** The time elapsed between releasing one key and pressing the next.

**Question 2 (Security Policy):**

**(a) Information Security Policy & Importance:**
    *   **Definition:** A set of written rules, practices, and procedures established by an organization to protect its information assets, ensuring their confidentiality, integrity, and availability, and guiding employee behavior.
    *   **Importance:** It provides a clear framework for security operations, defines acceptable behavior, helps ensure compliance with laws and regulations, reduces risk, demonstrates due diligence, and provides a basis for responding to security incidents.

**(b) Scenarios & Policies:**
    **(i) Sensitive printouts left unattended:** **Clean Desk Policy.** This policy would mandate that employees secure sensitive documents when away from their desks and at the end of the day, preventing unauthorized viewing or theft of physical information.
    **(ii) Remote employee on unsecured Wi-Fi:** **Remote Access Policy.** This policy would specify secure methods for connecting to the company network remotely (e.g., requiring VPN usage), types of approved networks, and security configurations for remote devices, thus protecting data in transit.
    **(iii) Malware via email links:** **Email Security Policy** (and/or Social Engineering Awareness Policy). The Email Security Policy would provide guidelines on identifying and handling suspicious emails, attachments, and links. A Social Engineering Awareness Policy would educate users on such tactics.

**(c) Unauthorized P2P Software:**
    **(i) Policy:** **Software Installation Policy** (or Acceptable Use Policy with a specific clause on software).
    **(ii) Key Elements/Rules:**
        1.  **Prohibition of Unauthorized Software:** Clearly state that installing software not approved by IT is forbidden.
        2.  **Approval Process:** Define a formal process for requesting and obtaining approval for any new software installation, including a security review.
        3.  **Regular Audits/Monitoring:** Implement mechanisms to scan company devices for unauthorized software installations and enforce compliance.
        4.  (Optional) Consequences: Clearly state the consequences for violating the policy.

---

This detailed breakdown should give you a very strong basis for revising these two important topics. Remember to refer back to your slides for exact phrasing and any additional examples provided by your lecturer. Good luck!




Okay, let's create targeted revision materials for **Physical Security Control** and **Protection Implemented related to Network Security** based on your BAIT1093 course slides.

---

## Part 1: Revision Notes

---

### **Topic 1: Physical Security Control**

(Primarily from Chapter 6)

**I. Overview and Purpose (Chapter 6, Slide 3)**

*   **Definition/Goal:** To protect physical assets that support the storage and processing of information.
*   **Involves Two Complementary Requirements:**
    1.  **Prevent damage to physical infrastructure:** This includes damage from natural disasters, accidents, or malicious acts.
    2.  **Prevent physical infrastructure misuse:** This aims to stop misuse that could lead to the compromise (misuse or damage) of protected information.
*   **Concerns include (Chapter 6, Slide 4):**
    *   **Information system hardware:** Servers, workstations, networking equipment, storage media.
    *   **Physical facility:** Buildings and structures housing systems.
    *   **Supporting facilities:** Electrical power, communication services, environmental controls (HVAC).
    *   **Personnel:** Humans involved in control, maintenance, and use of systems.
*   **Threats to Physical Security (Chapter 6, Slide 6):**
    *   Unauthorized access to company premises
    *   Theft (of equipment, by copying)
    *   Vandalism
    *   Fire
    *   Unstable power supply
    *   Humidity
    *   Natural disasters (Lightning, Floods, Earthquakes)

**II. External Perimeter Defenses (Chapter 6, Slides 7-9)**

*   **Purpose:** Restrict access to areas where equipment is located; keep intruders from entering a campus, building, or other area.
*   **Passive Barriers (Chapter 6, Slide 7):**
    *   **Fencing:** Tall, permanent structures.
    *   **Signage:** Explains restricted areas and warnings.
    *   **Proper Lighting:** Allows areas to be viewed after dark, deterring intruders and aiding surveillance.
*   **Active Security Elements (Chapter 6, Slide 8):**
    *   **Personnel (Human Security Guards):** Patrol and monitor restricted areas.
    *   **Video Surveillance Cameras (CCTV - Closed Circuit Television):** Transmit signals to a specific, limited set of receivers for monitoring activity.
*   **Sensors (Chapter 6, Slide 9):** Placed in strategic locations to alert guards by generating audible alarms for unexpected/unusual actions.
    *   **Motion detection:** Detects object's change in position (passive/active infrared).
    *   **Noise detection:** Detects suspicious noise (microphones with noise-activated tech).
    *   **Temperature detection:** Detects sudden temperature changes (thermal camera for lurking individuals).
    *   **Proximity:** Detects presence of an object (target) entering sensor's field (using sound, light, IR, electromagnetic fields).

**III. Internal Physical Security Controls (Chapter 6, Slides 10-18)**

*   **Purpose:** Second layer of defense if external perimeters are breached.
*   **Locks (Chapter 6, Slide 10):**
    *   Restrict access to doors, cabinets.
    *   Physical locks requiring keys or other devices are common.
    *   Servers, routers, switches, hubs should be in locked, secure rooms with limited access.
*   **Mantraps (Chapter 6, Slide 11):**
    *   Designed as an air gap to separate a nonsecure area from a secured area.
    *   Controls two interlocking doors; only one door can be open at any time.
    *   Monitors and controls passage, preventing tailgating.
*   **Environmental Controls (Chapter 6, Slide 12):**
    *   **Fire Prevention:**
        *   Automatic fire detectors.
        *   Extinguishers that do *not* use water (to protect electronics).
        *   Fireproof safes for backup tapes.
    *   **Power Supply:** Voltage controllers for unstable power.
    *   **Humidity Control:** Air conditioners for server/computer rooms.
*   **Protection from Lightning/Flood (Chapter 6, Slide 13):**
    *   **Lightning protection systems:** Reduce chances of lightning damage (not 100% perfect).
    *   **Flood protection:** Housing computer systems in high lands.
*   **Server Room/Data Center Specific Rules (Chapter 6, Slides 14-17):**
    *   **Compliance:** Design to industry standards (e.g., ISO 27001, NIST SPs, DoD frameworks).
    *   **Physical Structure:** Fire-resistant room, strong door with a strong lock (e.g., deadbolt).
    *   **HVAC:** Good Heating, Ventilation, and Air Conditioning system.
    *   **Access Control (Personnel):**
        *   Only necessary personnel have key access.
        *   Enforce multi-layer authentication (passwords, RFID, biometrics).
    *   **Logging:** Server room log (manual or automatic via electronic/biometric locks) for entry/exit.
    *   **Data Security:** Data stored in servers/data centers should be encrypted.
    *   **Redundancy:** Server systems designed for redundancy (alternative/redundant storage).
    *   **Emergency Services:** Automated and highly available access to police, healthcare, firefighting. Automated notification systems.
*   **Workstations/Laptops Security (Chapter 6, Slide 18):**
    *   Engraved identifying mark.
    *   Routine inventory.
    *   Attach to desks with cables (effective and affordable).

---

### **Topic 2: Protection Implemented related to Network Security**

(Primarily from Chapter 5; some elements from Chapter 3 for crypto context)

**I. Network Security Overview (Chapter 5, Slides 3-5)**

*   **Goal:** Protect confidential information on a network.
*   **Information States on a Network:**
    *   On physical storage media (at rest).
    *   In transit across the network in the form of packets (in motion) – *primary focus for network security issues*.
*   **Challenge:** Protecting networks connected to the internet from numerous unknown networks and potentially malicious users.
*   **Key Questions (Chapter 5, Slide 7):**
    *   How to protect confidential info from those without explicit access needs?
    *   How to protect network and resources from malicious users/accidents originating externally?

**II. Common Attacks on Networks (Chapter 5, Slides 8-23)**

*   **Network Packet Sniffers (Slides 9-14):**
    *   Capture and analyze network packets (often sent in clear text).
    *   Can reveal sensitive info (usernames, passwords), database queries, network topology.
    *   Exploits promiscuous mode of network adapter cards.
*   **IP Spoofing and Denial-of-Service (DoS) Attacks (Slides 15-17):**
    *   **IP Spoofing:** Attacker pretends to be a trusted computer by using an IP address from the trusted range or a trusted external IP.
    *   **DoS via IP Spoofing:** Attacker spoofs an IP and sends requests (e.g., TCP SYN). Targeted host tries to respond to the spoofed (often non-existent) IP, exhausting resources while waiting for handshake completion, becoming unable to serve legitimate requests.
*   **Password Attacks (Slides 18-19):** (Covered in detail in "Means of Authentication") Brute-force, dictionary attacks across the network to log into shared resources.
*   **Distribution of Sensitive Internal Information to External Sources (Slides 20-21):** Often by disgruntled employees or via compromised accounts.
*   **Man-in-the-Middle (MITM) Attacks (Slides 22-23):**
    *   Attacker intercepts communication between two parties.
    *   Requires access to network packets (e.g., ISP employee, compromised router).
    *   Can steal info, hijack sessions, analyze traffic, perform DoS, corrupt/inject data.

**III. Network Security Components (Defenses) (Chapter 5, Slides 24-47)**

*   **Malware Scanners (Slides 25-27):** (Also an endpoint security measure but relevant for network-borne threats)
    *   Prevent malware infection.
    *   *Signature-based:* Match against known malware definitions.
    *   *Behavior-based (Heuristic):* Look for malware-like behavior. Can have false positives/negatives.
    *   *Techniques:* Email/attachment scanning, download scanning, file scanning, heuristic scanning, sandbox (isolated execution), machine learning.
*   **Firewalls (Slides 28-37):**
    *   Barrier between networks or computers, filtering incoming/outgoing packets.
    *   *Benefits:* Block unwanted traffic based on rules, prevent DoS, prevent external scanning of internal network details.
    *   *Limitations:* Cannot block all attacks (e.g., user downloading Trojan), may not stop internal attacks.
    *   **Types:**
        *   **Stateless Packet Filtering (Slides 31-32):** Simplest. Examines each packet individually based on protocol, port, IP address. Susceptible to DoS (e.g., SYN floods) as it doesn't track connection state.
        *   **Stateful Packet Inspection (SPI) (Slides 33-34):** Examines packets in the context of the conversation (connection state). Aware of previous packets. Less susceptible to floods/spoofing. Can look at actual packet contents for advanced filtering. *Recommended type.*
        *   **Application Gateway (Proxy Firewall / Application-Level Proxy) (Slides 35-37):** Client connects to proxy; proxy connects to destination. Hides internal IPs. Acts on behalf of client. Makes decisions about which packets to forward based on application-level data. Often includes SPI.
*   **Antispyware (Slide 38):** Scans for and removes spyware. (Often part of broader antimalware).
*   **Intrusion Detection Systems (IDSs) & Intrusion Prevention Systems (IPSs) (Slides 39-42):**
    *   **IDS:** Inspects inbound/outbound port activity for patterns indicating break-in attempts or malicious activity (e.g., network scanning).
        *   *Passive IDS:* Monitors and logs suspicious activity, may notify admin.
        *   *Active IDS (IPS):* Takes action to block/shut down suspicious communication. Can have false positives (blocking legitimate traffic).
    *   Firewall controls traffic flow based on port/protocol; IPS analyzes packet content for malicious payloads or anomalous activity.
*   **Digital Certificates (Slide 43-44):**
    *   X.509 standard. Contains user's/server's public key and other info, signed by a trusted Certificate Authority (CA).
    *   Authenticates the holder of the certificate (e.g., proves a website is genuine).
    *   Used for secure communication (e.g., HTTPS).
*   **SSL/TLS (Secure Sockets Layer / Transport Layer Security) (Slide 45):**
    *   Protocols that provide secure communication (encryption, authentication) over a network, typically for web traffic (HTTPS).
    *   Use both asymmetric (for key exchange, authentication) and symmetric (for bulk data encryption) cryptography.
*   **Virtual Private Networks (VPNs) (Slide 46):**
    *   Creates a secure, encrypted "tunnel" over a public network (like the Internet) to connect a remote user/site to a central location.
    *   Emulates a direct private network connection.
*   **Wi-Fi Security (Wireless Network Security) (Slide 47 & Slides 61-70 in detail - see section IV below):**
    *   Protocols like WEP (insecure), WPA, WPA2, WPA3.

**IV. Wireless Network Security (Chapter 5, Slides 61-70)**

*   **Importance:** Growing number of users and devices rely on wireless connectivity.
*   **Wireless Attacks:**
    *   **Bluetooth Attacks (Slide 62):**
        *   *Bluejacking:* Sending unsolicited messages to Bluetooth-enabled devices (annoyance).
        *   *Bluesnarfing:* Accessing unauthorized information (contacts, emails, files) from a device via Bluetooth without owner's knowledge.
    *   **Near Field Communication (NFC) Attacks (Slides 63-64):** (Short-range, contactless payments)
        *   *Eavesdropping:* Intercepting unencrypted NFC communication.
        *   *Data theft:* "Bumping" a portable reader against a user's smartphone to steal NFC payment info.
        *   *Man-in-the-middle:* Intercepting and forging NFC communications.
        *   *Device theft:* Using a stolen (and unlocked) phone for purchases.
    *   **Radio Frequency Identification (RFID) Attacks (Slides 65-66):** (Employee badges, inventory tags)
        *   *Unauthorized tag access:* Rogue reader determines inventory/tracks sales.
        *   *Fake tags:* Undermine inventory integrity.
        *   *Eavesdropping:* Intercepting communications between RFID tags and readers.
    *   **Wireless Local Area Network (WLAN/Wi-Fi) Attacks (Slides 67-69):**
        *   *IEEE 802.11 standards define Wi-Fi versions.*
        *   **Rogue Access Point:** Unauthorized AP allowing attackers to bypass network security.
        *   **Evil Twin:** Fake AP mimicking a legitimate one to trick users into connecting, allowing interception.
        *   **Intercepting Wireless Data:** Capturing RF signals from open/misconfigured APs.
        *   **Wireless Denial of Service:** Flooding RF spectrum with noise to jam communications or disrupt device connection to AP.
*   **Wireless Security Solutions (Slide 70):**
    *   **Access control:** MAC address filtering (limiting which devices can connect to AP).
    *   **Strong Wi-Fi Security Protocols:** WPA3 or WPA2 (WEP is insecure). These protocols provide encryption for wireless traffic.
        *   *WEP (Wired Equivalent Privacy):* Old, many vulnerabilities, easily cracked. **NOT SAFE.**
        *   *WPA (Wi-Fi Protected Access):* Improvement over WEP, used TKIP.
        *   *WPA2 (Wi-Fi Protected Access II):* Stronger, uses AES-CCMP. **Commonly used today.**
        *   *WPA3:* Newest, enhanced security features (e.g., stronger encryption, protection against offline dictionary attacks, easier IoT onboarding). **Recommended.**
    *   **Site surveys:** For optimal AP placement ensuring adequate coverage, bandwidth, and security.

---

## Part 2: Flashcards

---

(All questions first, then all answers)

**Flashcard Questions: Physical Security Control**

101.Q: What are the two primary complementary requirements of physical security?
102.Q: Name three examples of physical security threats to a server room.
103.Q: What is the purpose of "proper lighting" as an external perimeter defense?
104.Q: Differentiate between a passive barrier and an active security element in external perimeter defenses.
105.Q: How does a motion detection sensor typically alert security personnel?
106.Q: What is a mantrap and how does it control access?
107.Q: Why should water-based extinguishers generally not be used in server rooms?
108.Q: List two specific physical security rules for server rooms regarding access by personnel.
109.Q: Name one physical security measure for protecting laptops in an office.

**Flashcard Questions: Protection Implemented related to Network Security**

110.Q: What is the primary goal of network security?
111.Q: How does a "network packet sniffer" compromise security?
112.Q: Explain how IP spoofing can lead to a Denial-of-Service (DoS) attack.
113.Q: What is a Man-in-the-Middle (MITM) attack?
114.Q: Differentiate between stateless packet filtering and stateful packet inspection (SPI) firewalls.
115.Q: What is an Application Gateway (Proxy Firewall)?
116.Q: What is the difference between an Intrusion Detection System (IDS) and an Intrusion Prevention System (IPS)?
117.Q: What is the main purpose of a Digital Certificate (e.g., X.509) in network security?
118.Q: What does the "S" in HTTPS signify and what protocols typically provide this?
119.Q: How does a VPN enhance network security when using the internet?
120.Q: What is an "Evil Twin" in the context of Wi-Fi attacks?
121.Q: Which Wi-Fi security protocol is considered highly insecure and should not be used?
122.Q: What is the most current and recommended Wi-Fi security protocol?

---

**Flashcard Answers**

*Physical Security Control Answers:*

101.A: 1. Prevent damage to physical infrastructure. 2. Prevent physical infrastructure misuse leading to compromise of protected information.
102.A: Fire, Flood/Water Damage, Unauthorized Access/Theft, Vandalism, Power Outage. (Any three)
103.A: To deter intruders by increasing visibility, and to aid surveillance (human guards or CCTV) after dark.
104.A: A passive barrier (e.g., fence, signage) physically obstructs or warns without active components. An active security element (e.g., security guard, CCTV) actively monitors or responds to threats.
105.A: By generating an audible alarm and/or sending a notification to security personnel or a central monitoring station.
106.A: A physical access control system with two interlocking doors where only one door can be open at a time. It controls entry/exit by preventing tailgating and ensuring individuals pass one by one.
107.A: Water can damage electronic equipment. Non-water-based (e.g., gas-based) extinguishers are preferred for server rooms.
108.A: 1. Only authorized personnel with a legitimate need should have key/access. 2. Enforce multi-layer authentication for entry. (Also: server room logs).
109.A: Attaching laptops to desks with security cables.

*Protection Implemented related to Network Security Answers:*

110.A: To protect the confidentiality, integrity, and availability of information and resources on a network.
111.A: It captures and analyzes network packets, potentially revealing sensitive information like usernames, passwords, and confidential data if the traffic is unencrypted (clear text).
112.A: An attacker sends packets with a spoofed (fake) source IP address to a target. The target attempts to respond to the spoofed IP, and if the spoofed IP doesn't respond (or is overwhelmed), the target's resources get consumed waiting for replies or managing half-open connections, leading to a DoS for legitimate users.
113.A: An attack where the attacker secretly intercepts and possibly alters communications between two parties who believe they are directly communicating with each other.
114.A: Stateless firewalls examine each packet individually based on source/destination IP, port, and protocol, without considering past traffic. Stateful (SPI) firewalls track the state of active connections and make decisions based on the context of the traffic stream, offering better security against attacks like spoofing and DoS.
115.A: A type of firewall that acts as an intermediary (proxy) for specific applications. It inspects traffic at the application layer, hiding internal network details and making decisions based on application-specific commands and data.
116.A: An IDS monitors network or system activities for malicious actions or policy violations and reports them. An IPS is an active system that also monitors and, upon detecting malicious activity, attempts to block or prevent it.
117.A: To bind a public key to an identity (a person, server, or organization) and authenticate that identity, typically vouched for by a trusted Certificate Authority (CA).
118.A: The "S" in HTTPS stands for "Secure." This security is typically provided by SSL (Secure Sockets Layer) or, more commonly today, TLS (Transport Layer Security) protocols, which encrypt the communication.
119.A: A VPN creates an encrypted, secure tunnel over a public network (like the internet) between a remote user/site and a private network, protecting data confidentiality and integrity from eavesdroppers on the public network.
120.A: A fraudulent Wi-Fi access point that appears to be a legitimate one, set up to eavesdrop on wireless communications or steal credentials from users who connect to it.
121.A: WEP (Wired Equivalent Privacy).
122.A: WPA3 (Wi-Fi Protected Access 3).

---

## Part 3: Mini-Tests

---

**Mini-Test 1: Physical Security Control (15 minutes)**

1.  **Multiple Choice (Choose one):** Which of the following is an example of an *active* external perimeter defense?
    a)  Warning Signage
    b)  High Fencing
    c)  CCTV System
    d)  Proper Lighting

2.  **Short Answer:** Explain the primary purpose of a mantrap in controlling physical access.

3.  **Scenario:** A company is setting up a new server room. List three distinct environmental controls they should consider implementing to protect their servers.

4.  **List:** Name two types of sensors that can be used to detect unauthorized presence in a restricted area.

**Mini-Test 1: Answers**

1.  c) CCTV System
2.  The primary purpose of a mantrap is to control the passage of individuals between a non-secure and a secure area by allowing only one person to pass at a time through a system of two interlocking doors, thus preventing tailgating and unauthorized entry.
3.  Three environmental controls:
    *   **Fire suppression system** (e.g., gas-based, not water).
    *   **HVAC (Heating, Ventilation, and Air Conditioning)** for temperature and humidity control.
    *   **Uninterruptible Power Supply (UPS)** or voltage controllers for stable power.
4.  Motion detection sensor, Noise detection sensor (or Temperature, Proximity).

---

**Mini-Test 2: Network Security Protections (15 minutes)**

1.  **True/False (Justify your answer):** A stateless packet filtering firewall is generally more effective against SYN flood attacks than a stateful packet inspection firewall.

2.  **Short Answer:** What is the main difference in functionality between an Intrusion Detection System (IDS) and an Intrusion Prevention System (IPS)?

3.  **Scenario:** A user receives an email that appears to be from their bank, asking them to click a link and log in to verify their account details. The link leads to a fake website. If the user was communicating with the real bank, what network security protocol (evident in the browser) would help ensure the communication is encrypted and the bank's identity is verified?

4.  **List:** Name three common types of attacks that target wireless networks specifically.

**Mini-Test 2: Answers**

1.  False. A stateful packet inspection (SPI) firewall is generally *more* effective against SYN flood attacks because it tracks the state of connections. It can recognize that SYN requests are not being completed with ACKs and can take action, whereas a stateless firewall treats each SYN packet individually without context.
2.  An IDS primarily *detects and alerts* on suspicious activity or attacks. An IPS *detects and actively attempts to block or prevent* the suspicious activity or attack.
3.  HTTPS (Hypertext Transfer Protocol Secure), which uses SSL/TLS (Secure Sockets Layer / Transport Layer Security). The browser would also show a lock icon and allow inspection of the bank's digital certificate.
4.  Evil Twin, Rogue Access Point, Wireless Denial of Service (or Bluejacking, Bluesnarfing, NFC attacks, RFID attacks).

---

## Part 4: Mock Exam Questions

---

**Question 1 (Physical & Network Security Concepts - 25 marks)**

**(a)** A company is designing the physical security for its new data center.
    **(i)** Describe three different types of *external perimeter defenses* they should implement, explaining the purpose of each. (6 marks)
    **(ii) Explain how "Multi-Layer Authentication" should be applied for access to the server rooms within the data center as per your course notes. (4 marks)

**(b)** Differentiate between a *stateless packet filtering firewall* and a *stateful packet inspection (SPI) firewall* in terms of how they process network traffic and their effectiveness against certain types of attacks. (6 marks)

**(c)** What is a "Man-in-the-Middle (MITM)" attack? Briefly describe one scenario where it might occur and one potential consequence. (5 marks)

**(d)** Explain the purpose of a Virtual Private Network (VPN) and how it provides security. (4 marks)

**Question 2 (Wireless & Network Security Components - 25 marks)**

**(a)** Wireless networks are ubiquitous but present unique security challenges.
    **(i)** Describe the "Evil Twin" attack. How does it deceive users? (4 marks)
    **(ii)** Which Wi-Fi security protocol (WEP, WPA, WPA2, WPA3) is considered obsolete due to severe vulnerabilities, and which is currently the most recommended for strong security? (3 marks)
    **(iii) Besides using strong encryption protocols, list two other measures that can enhance WLAN security. (3 marks)

**(b)** An Intrusion Detection System (IDS) and an Intrusion Prevention System (IPS) are critical network security components.
    **(i)** Explain the primary difference between a passive IDS and an active IDS (IPS). (4 marks)
    **(ii)** What is a "false positive" in the context of an IPS, and what could be a negative consequence of a false positive? (4 marks)

**(c)** Secure Sockets Layer (SSL) / Transport Layer Security (TLS) is fundamental for secure web communication.
    **(i)** What are the two main security services provided by SSL/TLS to a web session (e.g., HTTPS)? (4 marks)
    **(ii)** What role does a Digital Certificate play in establishing a secure SSL/TLS connection? (3 marks)

---

**Mock Exam Model Answers (Key Points)**

**Question 1:**

**(a) Data Center Physical Security:**
    **(i) External Perimeter Defenses:**
        1.  **Fencing:** A physical barrier to define the boundary and deter casual entry.
        2.  **Security Guards:** Active personnel to monitor, patrol, verify identities, and respond to incidents.
        3.  **CCTV Surveillance:** Video cameras to record activity, deter intruders, and provide evidence for investigations. (Also: Lighting, Signage, Motion Sensors).
    **(ii) Multi-Layer Authentication for Server Rooms:** This involves requiring more than one type of authentication factor for access. For server rooms, this could mean:
        *   Something you have (e.g., an authorized access card/badge - RFID).
        *   Something you know (e.g., a PIN entered on a keypad).
        *   Something you are (e.g., a biometric scan like fingerprint or iris at the server room door).
        This makes it significantly harder for an unauthorized person to gain access even if one factor is compromised.

**(b) Stateless vs. Stateful Firewalls:**
    *   **Stateless Packet Filtering Firewall:** Examines each network packet in isolation. Makes decisions based on information in the packet header (source/destination IP, port, protocol) against a static rule set. It does not maintain any context about active connections.
        *   *Effectiveness:* Can block known unwanted ports/IPs but is vulnerable to attacks that exploit connection states, like IP spoofing in DoS attacks (e.g., SYN floods), as it doesn't know if a packet is part of an established, legitimate session.
    *   **Stateful Packet Inspection (SPI) Firewall:** Tracks the state of active network connections (e.g., TCP connection states). It examines packet headers and also considers whether the packet legitimately belongs to an existing connection.
        *   *Effectiveness:* More secure. It can detect and block spoofed packets or those not part of a valid session, offering better protection against DoS attacks (like SYN floods) and other sophisticated attacks by understanding the context of the communication.

**(c) Man-in-the-Middle (MITM) Attack:**
    *   **Definition:** An attack where an adversary secretly positions themselves between two communicating parties, intercepting and potentially altering their communications. The parties believe they are communicating directly with each other.
    *   **Scenario:** An attacker sets up a rogue Wi-Fi access point (Evil Twin) in a public place. When users connect to it, the attacker can intercept all their unencrypted internet traffic.
    *   **Consequence:** Theft of sensitive information (e.g., login credentials, credit card details), injection of malware, or alteration of transmitted data.

**(d) VPN Purpose and Security:**
    *   **Purpose:** To provide a secure and private communication channel between a remote user/site and a central private network, by tunneling traffic over a public network like the Internet.
    *   **Security:** It provides security through:
        1.  **Encryption:** Data transmitted through the VPN tunnel is encrypted, protecting its confidentiality from eavesdroppers on the public network.
        2.  **Authentication:** VPNs often require authentication of users and devices to ensure only authorized entities can establish a connection.
        3.  (Often) **Integrity:** Mechanisms to ensure data is not tampered with during transit.

**Question 2:**

**(a) Wireless Security:**
    **(i) Evil Twin Attack:** An attacker sets up a fraudulent Wi-Fi Access Point that mimics the name (SSID) and potentially other characteristics of a legitimate, trusted AP (e.g., at a coffee shop or airport). It deceives users by tricking their devices into automatically connecting to the malicious AP, believing it's the legitimate one. Once connected, the attacker can intercept traffic, steal credentials, or launch other attacks.
    **(ii) Wi-Fi Protocols:**
        *   **Obsolete/Insecure:** WEP (Wired Equivalent Privacy).
        *   **Most Recommended:** WPA3 (Wi-Fi Protected Access 3). (WPA2 is still widely used and generally secure if configured properly).
    **(iii) Other WLAN Security Measures:**
        1.  **MAC Address Filtering:** Allowing only specific, registered device MAC addresses to connect to the AP.
        2.  **Disabling SSID Broadcast:** Making the network name not publicly visible (though easily discoverable by determined attackers). (Better: Strong Passphrases, Regular Firmware Updates for APs, Network Segmentation).

**(b) IDS/IPS:**
    **(i) Passive IDS vs. Active IDS (IPS):**
        *   **Passive IDS:** Monitors network traffic or system logs for suspicious activity or known attack signatures. When a potential threat is detected, it generates an alert or logs the event but does not take direct action to block it.
        *   **Active IDS (IPS):** Also monitors for threats but, upon detection, can automatically take action to block the malicious traffic or activity, such as terminating the connection or blocking the source IP address.
    **(ii) False Positive (IPS):**
        *   **Definition:** An event where the IPS incorrectly identifies legitimate, harmless network traffic or user activity as malicious and consequently blocks it.
        *   **Consequence:** Disruption of legitimate business operations or user access to necessary resources, loss of productivity, and frustration for users.

**(c) SSL/TLS:**
    **(i) Two Main Security Services:**
        1.  **Confidentiality:** Encrypts the data exchanged between the client (browser) and the server, preventing eavesdropping.
        2.  **Authentication:** Verifies the identity of the server (and optionally the client) to ensure the client is connecting to the intended, legitimate server and not an imposter. (Also provides Integrity).
    **(ii) Role of Digital Certificate:** The server presents its digital certificate to the client. This certificate, issued by a trusted Certificate Authority (CA), contains the server's public key and affirms the server's identity. The client's browser verifies the certificate's validity and authenticity (using the CA's public key), thus trusting the server's public key for the secure key exchange phase of TLS.

---

This should provide a good basis for your revision on these topics. Remember to cross-reference with your slides for specific details and examples emphasized in your course!




