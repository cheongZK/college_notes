Okay, I will help you solve the questions from these exam papers. Please note that for a real exam, your answers would need to be in your own words and understanding. I will provide detailed answers and explanations.

Let's go through them one by one, starting with the **October 2023 Examination (the second image set you provided in the first interaction, with Maybank2u in Q1).**

---

**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**
**FACULTY OF COMPUTING AND INFORMATION TECHNOLOGY**
# **ACADEMIC YEAR 2023/2024**
**OCTOBER EXAMINATION**
**INFORMATION TECHNOLOGY BAIT1093**
**INTRODUCTION TO COMPUTER SECURITY**

---

**Question 1**

**a) "Welcome to Maybank2u, Malaysia's no. 1 online banking site. Enjoy the convenience of online banking at anytime, anywhere with Maybank2u". This is the caption found on the Maybank online banking website. Provide examples of the THREE (3) security goals' requirements related to the system, thus showing the significance of the requirements. (12 marks)**

The three fundamental security goals (often called the CIA triad) are Confidentiality, Integrity, and Availability. Here's how they relate to Maybank2u:

1.  **Confidentiality:**
    *   **Requirement:** Ensuring that sensitive banking information (account balances, transaction history, personal identification details, login credentials) is accessible only to authorized users (the account holder). It prevents unauthorized disclosure of information.
    *   **Example for Maybank2u:**
        *   **Encryption:** Maybank2u must use strong encryption (e.g., HTTPS/TLS) for all data transmitted between the user's browser and the bank's servers. This prevents eavesdroppers from intercepting and reading login credentials or financial data.
        *   **Access Controls:** Only the authenticated user can view their own account details. A user cannot see another user's balance or transaction history.
    *   **Significance:** If confidentiality is breached, it could lead to identity theft, financial fraud, loss of customer trust, and severe reputational damage for the bank.

2.  **Integrity:**
    *   **Requirement:** Ensuring the accuracy and completeness of information and processing methods. Data should not be improperly modified, either accidentally or maliciously. Transactions must be processed correctly.
    *   **Example for Maybank2u:**
        *   **Transaction Validation:** When a user initiates a fund transfer, Maybank2u must ensure that the amount specified is the amount actually debited from the sender's account and credited to the recipient's account, without any unauthorized alteration during the process.
        *   **Data Hashing/Checksums:** Internal systems might use hashing to ensure that stored account balances or transaction records haven't been tampered with. Digital signatures on transaction confirmations can also ensure integrity.
    *   **Significance:** A breach of integrity could lead to incorrect account balances, fraudulent transactions being processed, or legitimate transactions being altered, causing direct financial loss to customers or the bank, and eroding trust.

3.  **Availability:**
    *   **Requirement:** Ensuring that authorized users have timely and reliable access to information and services when needed. The Maybank2u system must be operational and accessible for legitimate users to perform their banking activities "at anytime, anywhere."
    *   **Example for Maybank2u:**
        *   **Redundant Systems:** Maybank must have redundant servers, network connections, and power supplies to prevent single points of failure that could take the Maybank2u service offline.
        *   **Denial-of-Service (DoS) Protection:** Measures must be in place to protect against DoS attacks that could overwhelm the servers and make the service unavailable to genuine users.
    *   **Significance:** If Maybank2u is unavailable, customers cannot check balances, pay bills, or transfer funds, leading to significant inconvenience, potential financial penalties (e.g., late payment fees for bills), loss of business for the bank, and damage to its reputation as a reliable service provider.

**b) Demonstrate the THREE (3) elements for which trust management depends on with examples. (9 marks)**

Trust management in a system like online banking depends on:

1.  **Security Policy:**
    *   **Demonstration:** This is the set of rules and practices that dictate how an organization manages, protects, and distributes sensitive information. It defines what is considered acceptable behavior and what is not.
    *   **Example for Maybank2u:**
        *   Maybank's security policy would dictate password complexity requirements (e.g., minimum length, character types), session timeout rules, requirements for using multi-factor authentication (MFA) for certain transactions, and procedures for handling suspected fraudulent activity. It would also outline data retention and privacy policies. Users trust that the bank has and enforces a strong security policy to protect their assets and data.

2.  **Security Mechanisms:**
    *   **Demonstration:** These are the technical tools, controls, and procedures implemented to enforce the security policy. They are the practical application of the policy's rules.
    *   **Example for Maybank2u:**
        *   **Multi-Factor Authentication (MFA):** Requiring a one-time password (OTP) sent via SMS or generated by a secure app in addition to a username and password.
        *   **Encryption (TLS/SSL):** Using secure protocols to encrypt data in transit.
        *   **Firewalls and Intrusion Detection/Prevention Systems (IDS/IPS):** To protect the bank's network and servers from unauthorized access and malicious attacks.
        *   **Regular Security Audits and Penetration Testing:** To identify and fix vulnerabilities.
        Users trust that these mechanisms are robust, correctly implemented, and maintained to effectively protect the system.

3.  **Security Assurance:**
    *   **Demonstration:** This is the confidence or guarantee that the security policy and mechanisms are appropriate, implemented correctly, and consistently enforced. It involves verification, validation, and monitoring.
    *   **Example for Maybank2u:**
        *   **Third-Party Certifications/Audits:** Maybank might obtain certifications like ISO 27001 or undergo regular audits by independent security firms to validate their security posture. Displaying trust seals or compliance badges.
        *   **Transparency Reports/Security Information:** Providing customers with clear information about their security practices, how their data is protected, and what to do in case of a security incident.
        *   **Incident Response Plan:** Having a well-tested plan to deal with security breaches, assuring customers that issues will be handled effectively.
        Users gain assurance (and thus trust) from these visible commitments and verifications of the bank's security practices.

**c) "Security can be no stronger than its weakest link." Apply your knowledge to show how true is this statement with an example. (4 marks)**

This statement is profoundly true in computer security. It means that no matter how strong some security measures are, the overall security of a system is determined by its most vulnerable component. Attackers will always target the easiest point of entry or the most vulnerable part of a system.

**Application and Example:**
Consider an online banking system like Maybank2u that has implemented:
*   State-of-the-art firewalls.
*   Advanced intrusion detection systems.
*   Strong encryption for data transmission and storage.
*   Robust multi-factor authentication for user logins.

However, let's say the **weakest link** is **poor employee training on phishing awareness**.
An employee with access to administrative systems or customer data receives a convincing phishing email. Believing it to be legitimate, the employee clicks a malicious link and enters their credentials on a fake login page, or downloads and runs an attachment containing malware.

**How this shows the statement is true:**
Despite all the expensive and advanced technological security measures (firewalls, encryption, MFA on the *customer-facing* side), the attacker bypassed them by exploiting the human element—the poorly trained employee. This single weakness (the human vulnerability to social engineering) compromised the overall security of potentially sensitive internal systems or customer data. The strength of the firewalls or encryption becomes irrelevant if an attacker can gain legitimate credentials or install malware through a tricked employee. The employee's susceptibility to phishing was the weakest link, and it dictated the actual level of security.

---

**Question 2**

**a) Apply Caeser cipher to decrypt the given message: WKHB ZRQ. (8 marks)**

The Caesar cipher is a simple substitution cipher where each letter in the plaintext is shifted a certain number of places down or up the alphabet. To decrypt, we need to find the correct shift. We'll try common shifts, or assume a standard shift (like 3, which is common for Caesar cipher examples, but we need to test).

Let's try shifting backwards (left shift for decryption):
Original: W K H B Z R Q

Shift -1: V J G A Y Q P
Shift -2: U I F Z X P O
Shift -3: T H E Y W O N

The message "THEY WON" makes sense.
So, the key (shift) is 3. (Or -3 for decryption if considering a positive shift for encryption).

**Steps for decryption with a shift of -3:**
W (-3) -> T
K (-3) -> H
H (-3) -> E
B (-3) -> Y (B -> A -> Z -> Y)

Z (-3) -> W
R (-3) -> O
Q (-3) -> N

**Decrypted Message: THEY WON**

**b) Show the differences between block cipher and stream cipher. (8 marks)**

| Feature           | Block Cipher                                                                 | Stream Cipher                                                                    |
| :---------------- | :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| **Unit of Data**  | Encrypts data in fixed-size blocks (e.g., 64 bits, 128 bits).                | Encrypts data bit by bit or byte by byte, in a continuous stream.              |
| **Operation**     | If plaintext is larger than a block, it's divided. Padding may be needed for the last block. Operates in various modes (ECB, CBC, CTR, etc.). | Generates a pseudorandom keystream which is XORed with the plaintext stream.   |
| **Memory Usage**  | Generally requires more memory to hold a block of data.                      | Typically requires less memory as it processes smaller units.                    |
| **Speed**         | Can be slower, especially if complex modes of operation are used.            | Generally faster due to simpler operations on smaller units of data.             |
| **Error Propagation** | Depends on the mode of operation. Some modes (like CBC) cause error propagation within a block or to the next. | Errors in the ciphertext typically affect only the corresponding bit/byte in the plaintext upon decryption (if keystream is not affected). |
| **Usage**         | Secure storage of data, database encryption, file encryption (e.g., AES, DES). | Real-time communications like encrypted voice calls (VoIP), SSL/TLS for web traffic (e.g., RC4 (though deprecated), ChaCha20). |
| **Padding**       | Often requires padding if the plaintext length is not a multiple of the block size. | Does not typically require padding as it processes data as a stream.           |
| **Example Ciphers**| AES (Advanced Encryption Standard), DES (Data Encryption Standard), Blowfish.  | RC4 (Rivest Cipher 4 - now considered insecure for many uses), ChaCha20, Salsa20. |

**c) The two basic building blocks of all encryption techniques are substitution and transposition. Show the differences between the two techniques. (5 marks)**

1.  **Substitution Technique:**
    *   **Principle:** In a substitution cipher, letters or groups of letters in the plaintext are replaced by other letters, numbers, or symbols according to a defined scheme. The identity of the characters changes, but their positions remain the same.
    *   **How it works:** Each unit of plaintext (e.g., a character) is mapped to a corresponding unit of ciphertext.
    *   **Example:** The Caesar cipher (where 'A' becomes 'D', 'B' becomes 'E', etc., with a shift of 3). Another example is a simple monoalphabetic substitution where 'A' might be 'Q', 'B' might be 'X', etc.
    *   **Effect:** Obscures the relationship between the ciphertext and the original plaintext by changing the characters themselves.

2.  **Transposition Technique:**
    *   **Principle:** In a transposition cipher, the characters in the plaintext are rearranged or permuted. The characters themselves remain unchanged, but their order or position is altered.
    *   **How it works:** The plaintext letters are shuffled according to a specific rule or key.
    *   **Example:** A simple columnar transposition cipher where plaintext is written into rows of a grid and read off column by column. For instance, "HELLO WORLD" written in 2 rows:
        H L O W L
        E L O R D
        Reading columns might give: "HEL LL OW RO LD" (if read by column in order) or a permuted order if columns are permuted. A rail fence cipher is another example.
    *   **Effect:** Obscures the relationship between ciphertext and plaintext by scrambling the order of the characters, making it difficult to read.

**Key Difference:** Substitution changes the *identity* of the characters, while transposition changes their *position*. Modern ciphers (like AES) often use multiple rounds that combine both substitution and transposition techniques (e.g., in S-boxes for substitution and permutation layers for transposition) to create strong encryption.

**d) Provide TWO (2) requirements for secure use of symmetric encryption. (4 marks)**

1.  **Strong Key:**
    *   **Requirement:** The secret key used for both encryption and decryption must be cryptographically strong. This means it should be sufficiently long to resist brute-force attacks (e.g., for AES, keys of 128, 192, or 256 bits are standard) and truly random (or pseudorandom and unpredictable).
    *   **Significance:** The entire security of symmetric encryption relies on the secrecy and strength of this shared key. If the key is weak or guessed, the encryption is broken.

2.  **Secure Key Exchange/Distribution:**
    *   **Requirement:** The shared secret key must be exchanged between the sender and receiver in a secure manner, ensuring that no unauthorized third party can intercept or obtain the key.
    *   **Significance:** If an attacker obtains the key during distribution, they can decrypt all messages encrypted with that key. This is a major challenge in symmetric cryptography, often solved by using asymmetric cryptography for key exchange (e.g., Diffie-Hellman key exchange or encrypting the symmetric key with the receiver's public key).

---
Let's proceed to the next exam paper if you'd like, or I can continue with this one.
It seems you have provided multiple full exam papers. I will continue with the **October 2023** paper first.

---

**Question 3**

**a) John is a staff of ABC company, he is going to remote logon to the company's computer X.**

**(i) Demonstrate FIVE (5) steps to hijack John's TCP connection with X. (10 marks)**

TCP hijacking, also known as session hijacking, involves an attacker taking control of an established TCP session between two parties (John and computer X). Here are five steps an attacker might take:

1.  **Session Monitoring/Sniffing:**
    *   **Demonstration:** The attacker first needs to monitor the network traffic between John and computer X. This can be done if the attacker is on the same local network (e.g., using ARP spoofing to become a man-in-the-middle) or has compromised a router or switch along the communication path. The attacker sniffs packets to observe an active TCP session, looking for sequence numbers and acknowledgment numbers.
    *   **Goal:** To identify an active TCP session and gather critical information like IP addresses, port numbers, and current TCP sequence (SEQ) and acknowledgment (ACK) numbers being used by John and computer X.

2.  **Sequence Number Prediction/Synchronization:**
    *   **Demonstration:** TCP uses sequence numbers to ensure reliable, ordered delivery of data. To inject malicious packets into the session, the attacker must predict or determine the correct next sequence number that computer X is expecting from John, and the sequence number John is expecting from X.
    *   **Goal:** To craft packets that will be accepted as legitimate by either John or computer X by using the correct SEQ/ACK numbers.

3.  **Desynchronizing John's Connection (Optional but common):**
    *   **Demonstration:** The attacker might send a TCP RST (reset) packet or a flood of packets to John (appearing to come from computer X) or cause a denial-of-service against John. This temporarily prevents John from sending legitimate packets or makes his system unresponsive to computer X's packets.
    *   **Goal:** To prevent John from interfering with the hijacked session by sending conflicting packets or resetting the connection once the attacker takes over. If John sends packets with the "correct" sequence numbers after the attacker has used them, it can disrupt the hijacked session.

4.  **Injecting Malicious Packets/Taking Over the Session:**
    *   **Demonstration:** The attacker crafts packets with the correct predicted sequence numbers, spoofing John's IP address. These packets contain malicious commands or data intended for computer X. For example, if John was logged into a shell on computer X, the attacker could send commands as John.
    *   **Goal:** To send commands or data to computer X, impersonating John, and effectively take control of John's side of the session. Computer X believes these packets are legitimately from John.

5.  **Maintaining the Hijacked Session (and potentially blocking John's responses to X):**
    *   **Demonstration:** Once the attacker has injected packets, they need to continue predicting sequence numbers for ongoing communication. They also need to ensure that computer X's responses are either intercepted by them (if they are a man-in-the-middle) or that John's system doesn't send RST packets to computer X when it receives unexpected ACK numbers from X (in response to the attacker's injected packets).
    *   **Goal:** To continue interacting with computer X as John, potentially exfiltrating data or causing further damage, while preventing the legitimate John from regaining control or an RST packet from X terminating the attacker's connection.

**(ii) Provide ONE (1) countermeasure to prevent TCP hijacking. (3 marks)**

**Countermeasure: Use Encrypted Sessions (e.g., SSH, SSL/TLS, VPNs)**

*   **Explanation:** Implementing end-to-end encryption for the remote logon session significantly mitigates TCP hijacking.
    *   **SSH (Secure Shell):** For remote command-line access, SSH encrypts all traffic, including authentication credentials and the data exchanged. This prevents an attacker from sniffing useful information like sequence numbers or the content of the session. Even if an attacker could inject packets, they wouldn't know the encryption keys, so the injected data would be gibberish to the receiving end, and the session would likely terminate due to cryptographic errors.
    *   **SSL/TLS:** If the remote logon is via a web interface, ensuring it uses HTTPS (HTTP over SSL/TLS) provides similar protection.
    *   **VPN (Virtual Private Network):** John could connect to the company network via a VPN. The VPN creates an encrypted tunnel for all traffic between John's computer and the company network, protecting the TCP session from sniffing and straightforward injection.

**b) Explain message replay and provide THREE (3) countermeasures to prevent it. (12 marks)**

**Explanation of Message Replay (Replay Attack):**
A message replay attack is a form of network attack in which a valid data transmission is maliciously or fraudulently repeated or delayed. The attacker intercepts a message (or a sequence of messages) from a legitimate communication between two parties and then retransmits it later. The replayed message might be used to impersonate one of the parties, gain unauthorized access, or cause a system to perform an operation multiple times (e.g., replaying a financial transaction to withdraw money multiple times). The system receiving the replayed message believes it is a new, legitimate message because it was originally valid (e.g., correctly encrypted or signed).

**Example:**
John sends an encrypted and authenticated message to a server: "TRANSFER $100 TO ATTACKER_ACCOUNT". An attacker intercepts this valid message. Later, the attacker resends this *exact same* intercepted message to the server. If the server has no protection against replays, it will process the message again, transferring another $100.

**THREE (3) Countermeasures to Prevent Message Replay:**

1.  **Timestamps:**
    *   **How it works:** The sender includes a current timestamp in each message. The receiver checks the timestamp upon message arrival. If the timestamp is too old (outside an acceptable, pre-defined time window or "freshness" period), the message is rejected as a potential replay. Both sender and receiver must have synchronized clocks.
    *   **Benefit:** Prevents the replay of old messages.
    *   **Limitation:** Requires clock synchronization. A small window might reject legitimate messages in high-latency networks, while a large window might allow some replays.

2.  **Sequence Numbers (Nonces or Message Counters):**
    *   **How it works:** Each message sent in a session contains a unique, monotonically increasing sequence number (or a nonce, which is a "number used once"). The receiver keeps track of the last valid sequence number (or a list of recently received nonces). If a message arrives with a sequence number that is less than or equal to the last received valid one (or a nonce that has already been seen), it is rejected as a replay.
    *   **Benefit:** Ensures each message is processed only once within a session or context. Does not strictly rely on time synchronization (though can be combined with it).
    *   **Limitation:** Requires state management at the receiver to track sequence numbers/nonces. If the sequence number space is too small, it might wrap around.

3.  **Challenge-Response Authentication (One-Time Passwords/Tokens):**
    *   **How it works:** Before accepting a critical message or establishing a session, the receiver (server) sends a unique, unpredictable challenge (e.g., a random number) to the sender (client). The client must then incorporate this challenge into its response (e.g., by hashing it with a secret key or encrypting it). The server verifies the response. Since the challenge is unique for each attempt, a previously recorded response will not be valid for a new challenge. This ensures the sender is "live" and not replaying an old message. This is often seen in authentication protocols.
    *   **Benefit:** Very effective against replay as each transaction requires a fresh, unique response based on a new challenge.
    *   **Limitation:** Adds an extra round trip for the challenge-response, which can introduce latency. More complex to implement than simple timestamps or sequence numbers for every message in a stream.

---

**Question 4**

**a) Explain IT security management and provide its EIGHT (8) functions. (13 marks)**

**Explanation of IT Security Management:**
IT Security Management is a comprehensive and ongoing process within an organization aimed at identifying, assessing, managing, and mitigating risks to its information technology assets (hardware, software, data, networks) to an acceptable level. It involves establishing and maintaining a security framework that includes policies, procedures, standards, guidelines, and controls to protect the confidentiality, integrity, and availability (CIA) of IT systems and information. The goal is to ensure business continuity, minimize business damage by preventing and minimizing the impact of security incidents, and comply with legal, regulatory, and contractual requirements.

**EIGHT (8) Functions of IT Security Management:**

1.  **Policy Development and Maintenance:**
    *   Creating, reviewing, and updating security policies, standards, and procedures that define acceptable and unacceptable use of IT assets, security responsibilities, and control requirements (e.g., password policy, data backup policy, incident response policy).

2.  **Risk Management (Assessment and Treatment):**
    *   Identifying potential threats to IT assets, assessing vulnerabilities, analyzing the likelihood and impact of these threats materializing, and then implementing appropriate controls (risk treatment) to mitigate, transfer, accept, or avoid the risks.

3.  **Asset Management:**
    *   Identifying and cataloging all IT assets (hardware, software, information) and assigning ownership. Understanding the value of these assets is crucial for prioritizing security efforts.

4.  **Security Awareness and Training:**
    *   Educating employees, contractors, and other users about security policies, potential threats (like phishing, malware), their responsibilities in protecting IT assets, and how to respond to security incidents.

5.  **Access Control Management:**
    *   Implementing and managing mechanisms (e.g., usernames/passwords, multi-factor authentication, authorization rules) to ensure that users only have access to the information and system resources necessary for their job functions (principle of least privilege).

6.  **Incident Management and Response:**
    *   Developing and implementing procedures to detect, report, analyze, contain, eradicate, and recover from security incidents. This also includes post-incident analysis to learn lessons and improve defenses.

7.  **Compliance Management:**
    *   Ensuring adherence to relevant laws (e.g., data privacy laws like GDPR, PDPA), regulations (e.g., industry-specific regulations like PCI DSS for payment cards), and contractual obligations related to information security.

8.  **Monitoring, Auditing, and Review:**
    *   Continuously monitoring security logs and events for suspicious activity, conducting regular security audits (internal and external) to verify compliance and effectiveness of controls, and periodically reviewing the overall security posture to adapt to new threats and business changes.

**b) Show the differences between change management and configuration management. (9 marks)**

| Feature                       | Change Management                                                                                                | Configuration Management                                                                                                  |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **Primary Goal**              | To control and coordinate all changes to IT services and infrastructure in a planned and authorized manner, minimizing risk and disruption. | To identify, document, control, and verify the functional and physical characteristics of IT assets (Configuration Items - CIs) and their relationships. |
| **Focus**                     | The *process* of making changes: planning, approval, implementation, and review of changes.                    | The *state* of IT assets: tracking what assets exist, their versions, how they are configured, and how they interrelate. |
| **Scope**                     | Addresses all types of changes, including those to hardware, software, processes, documentation, and services.   | Primarily concerned with the configuration of specific IT components (CIs) and maintaining an accurate record of these configurations. |
| **Key Activities**            | Requesting changes, assessing impact and risk, obtaining authorization, scheduling, building/testing, implementing, and reviewing changes. | Identifying CIs, establishing baselines, controlling changes to configurations, status accounting (reporting), and verification/auditing of CIs. |
| **Trigger**                   | A Request for Change (RFC) or a perceived need for alteration or improvement.                                   | The lifecycle of an IT asset; a need to understand and control the components of the IT environment.                   |
| **Output/Deliverable**        | Implemented changes, updated documentation, post-implementation reviews, reduced adverse impact from changes.    | A Configuration Management Database (CMDB) or system (CMS), configuration baselines, audit reports, status reports.     |
| **Relationship to Each Other** | Change Management *uses* information from Configuration Management (e.g., to assess the impact of a change on specific CIs). Configuration Management is *updated by* the Change Management process (e.g., when a change to a CI is implemented, its record in the CMDB is updated). | Configuration Management provides the detailed information needed to make informed decisions within the Change Management process. |
| **Example**                   | A process for approving and deploying a security patch to all company servers, including testing and rollback plans. | Maintaining a database that details each server's hardware components, installed OS version, patch levels, installed applications, and network connections. |

**c) Explain the goal of IT security plan. (3 marks)**

The primary goal of an IT security plan is to **provide a strategic and tactical roadmap for an organization to protect its information assets (data, systems, networks) from threats and vulnerabilities, thereby ensuring the confidentiality, integrity, and availability of these assets.**

More specifically, it aims to:
1.  **Align security efforts with business objectives:** Ensuring that security initiatives support the overall mission and goals of the organization.
2.  **Manage risks effectively:** Identify, assess, and mitigate security risks to an acceptable level.
3.  **Guide resource allocation:** Provide a basis for prioritizing security investments and allocating resources (budget, personnel, technology) appropriately.
4.  **Ensure compliance:** Help the organization meet legal, regulatory, and contractual security obligations.
5.  **Facilitate coordinated security efforts:** Document roles, responsibilities, policies, procedures, and controls to ensure a consistent and comprehensive approach to security across the organization.

---
This completes the October 2023 paper. Let me know which paper you'd like to tackle next. I see "January 2024", "May/June 2024", and "May/June 2023" in the other images. Let's go in chronological order of examination if that works, or you can specify.

Based on the files, the next one after the initial TSMC paper seems to be the **January 2024 Examination (the third image set from the first interaction).**

---

**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**
**FACULTY OF COMPUTING AND INFORMATION TECHNOLOGY**
# **ACADEMIC YEAR 2023/2024**
**JANUARY EXAMINATION**
**INFORMATION TECHNOLOGY BAIT1093**
**INTRODUCTION TO COMPUTER SECURITY**

---

**Question 1**

**a) Describe TWO (2) types of attack vectors that may be used by a threat actor to penetrate a system. (10 marks)**

An attack vector is a path or means by which a threat actor can gain unauthorized access to a computer or network to deliver a payload or malicious outcome.

1.  **Phishing (Social Engineering via Email/Messaging):**
    *   **Description:** This is a social engineering technique where the attacker crafts deceptive emails, instant messages, or websites that appear to be from legitimate and trustworthy sources (e.g., banks, IT departments, known colleagues, popular online services). The goal is to trick the victim into performing an action that compromises their security.
    *   **How it's used to penetrate a system:**
        *   **Credential Theft:** The phishing message might contain a link to a fake login page (e.g., a fake bank login or company portal login). If the victim enters their username and password, the attacker captures these credentials. The attacker can then use these legitimate credentials to log into the real system, effectively penetrating it.
        *   **Malware Delivery:** The phishing email might contain a malicious attachment (e.g., a Word document with a macro virus, a PDF with embedded malware, or a ZIP file containing an executable). If the victim opens the attachment, malware (like a trojan, ransomware, or spyware) is installed on their system. This malware can then provide the attacker with remote access, exfiltrate data, or further compromise the system and network.
    *   **Impact:** Successful phishing can lead to unauthorized system access, data breaches, financial loss, and installation of various forms of malware.

2.  **Exploitation of Software Vulnerabilities (e.g., Unpatched Systems):**
    *   **Description:** Software vulnerabilities are flaws or weaknesses in the design, implementation, or configuration of software (operating systems, applications, web browsers, firmware). Threat actors actively search for and develop exploits – code that takes advantage of these vulnerabilities.
    *   **How it's used to penetrate a system:**
        *   **Remote Code Execution (RCE):** An attacker might find a vulnerability in a web server software or a network service running on a target system. By sending specially crafted data or requests to the vulnerable service, the attacker can trigger the flaw and execute arbitrary code on the target system, often with the privileges of the vulnerable service. This gives them a foothold.
        *   **Privilege Escalation:** If an attacker gains initial low-privilege access (e.g., through a compromised user account), they might then exploit a local vulnerability in the operating system or an application to gain higher privileges (e.g., administrator or root access), allowing full control over the system.
        *   **Drive-by Downloads:** A user visiting a compromised website (or a malicious website) that hosts an exploit kit can have malware silently installed on their system if their browser or browser plugins have unpatched vulnerabilities.
    *   **Impact:** Exploiting vulnerabilities can lead to complete system compromise, installation of backdoors, data theft, disruption of services, and use of the compromised system to launch further attacks. This is why timely patching of software is critical.

**b) Distinguish a passive attack from an active attack. (4 marks)**

| Feature             | Passive Attack                                                                | Active Attack                                                                               |
| :------------------ | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Nature of Action** | The attacker monitors or eavesdrops on communications or system activity without altering data or system resources. The goal is to obtain information. | The attacker attempts to alter system resources, affect their operation, or change data. The goal is to disrupt, modify, or damage. |
| **System Impact**   | Does not affect system resources or operations. The system continues to function normally. | Directly affects system resources or operations. May involve modification of data, denial of service, or unauthorized access. |
| **Detection**       | Difficult to detect because they do not change data or system state. Prevention is often the focus. | Easier to detect because they involve some modification or disruption that can be logged or observed. Detection and response are key. |
| **Primary Goal**    | Information gathering, reconnaissance, learning about the target.             | Disruption of services, unauthorized access, data modification, system damage, impersonation. |
| **Examples**        | Eavesdropping on network traffic (sniffing), traffic analysis (observing communication patterns even if encrypted). | Denial-of-Service (DoS) attacks, Man-in-the-Middle (MitM) attacks, data modification, malware infection, session hijacking, spoofing. |

**c) Ransomware is the largest growing types of malware and became widespread around 2010.**

**(i) Explain ransomware. (3 marks)**
Ransomware is a type of malicious software (malware) that, upon infecting a victim's computer or network, encrypts their files or locks them out of their system. The attackers then demand a ransom payment, typically in cryptocurrency (like Bitcoin) to remain anonymous, in exchange for providing a decryption key or restoring access. Some ransomware variants also exfiltrate sensitive data and threaten to publish it if the ransom is not paid (this is known as double extortion).

**(ii) Provide TWO (2) possible causes of a successful ransomware attack on a company. (6 marks)**

1.  **Successful Phishing Campaigns and Social Engineering:**
    *   **Cause:** Employees are often the weakest link. Attackers craft convincing phishing emails with malicious attachments (e.g., infected documents, ZIP files) or links to websites that download ransomware. If an employee opens the attachment or clicks the link and downloads the malware, the ransomware can then execute and spread through the company's network.
    *   **Why it leads to success:** Lack of security awareness training, employees being too busy or not cautious enough, or highly sophisticated and targeted phishing emails (spear phishing) can easily trick users into initiating the infection.

2.  **Unpatched Software Vulnerabilities:**
    *   **Cause:** Companies may fail to regularly update and patch their operating systems, applications (e.g., web browsers, office suites), and network devices. Attackers can exploit known vulnerabilities for which patches are available but not yet applied.
    *   **Why it leads to success:** Ransomware like WannaCry famously exploited an unpatched vulnerability in Windows (EternalBlue) to spread rapidly across networks without any user interaction. If systems are not patched, they remain open to automated exploits that can deliver and execute ransomware.

**(iii) List TWO (2) computer system strategies that can be adopted by companies to mitigate against ransomware attacks. (2 marks)**

1.  **Regular Data Backups and Offline/Immutable Storage:** Maintain frequent backups of critical data and ensure that at least one copy is stored offline (air-gapped) or in immutable storage (cannot be altered or deleted by ransomware).
2.  **Patch Management and Vulnerability Scanning:** Implement a robust patch management program to promptly apply security updates to operating systems, software, and firmware. Regularly scan for vulnerabilities.

---
I'll proceed with Question 2 of the January 2024 paper in the next message.Okay, continuing with the **January 2024 Examination**.

---

**Question 2**

**a) Assume you are the IT Security Officer of S&W Sdn Bhd. Recently, you are asked to prepare a data redundancy plan in case of any physical security breach to their data centres that may cause data loss.**

**(i) Identify THREE (3) categories of physical security threats that may cause data loss due to a physical security breach in S&W Sdn Bhd's data centres. (3 marks)**

1.  **Environmental Threats:** These are threats related to the natural or man-made environment surrounding and within the data centre.
2.  **Human Threats (Malicious or Accidental):** These are threats caused by human actions, whether intentional (e.g., sabotage, theft) or unintentional (e.g., errors, accidents).
3.  **Technical/Equipment Failures:** These are threats related to the failure of physical equipment or supporting infrastructure within the data centre.

**(ii) Give ONE (1) example for each category of physical security threat that you have listed in Question 2 a) (i). (3 marks)**

1.  **Environmental Threat Example:** **Fire** (e.g., due to electrical fault, arson, or nearby wildfire spreading) that could destroy servers and storage media.
2.  **Human Threat Example:** **Theft of hardware** (e.g., an unauthorized individual gaining access and stealing servers, hard drives, or backup tapes containing data).
3.  **Technical/Equipment Failure Example:** **HVAC (Heating, Ventilation, and Air Conditioning) system failure** leading to overheating, which can damage servers and cause data corruption or loss.

**(iii) Provide ONE (1) method that S&W Sdn Bhd should add in the data redundancy plan to prepare for data recovery when there is breach of physical security that you have listed in Question 2 a) (ii). (3 marks)**

**Method: Offsite Data Backups and Replication.**

*   **Explanation:** S&W Sdn Bhd should implement a strategy where data is regularly backed up to a secure, geographically separate offsite location. This could involve:
    *   **Regular Backups to Offsite Storage:** Periodically (e.g., daily) backing up critical data to tapes, disks, or a cloud backup service, with the physical media stored securely at a different location, or the cloud service being geographically distinct.
    *   **Data Replication:** For more critical systems requiring faster recovery, data can be replicated in near real-time to a secondary data centre or a cloud disaster recovery site.
*   **Relevance to Listed Threats:** If a fire destroys the primary data centre, or hardware is stolen, or an HVAC failure damages equipment, the data can be restored from these offsite backups or the replicated site, ensuring business continuity and minimizing data loss.

**(iv) Suggest FOUR (4) physical security prevention and mitigation measures that can be taken by S&W Sdn Bhd for data centres to prevent any data loss. (8 marks)**

1.  **Strict Access Control Systems:**
    *   **Measure:** Implement multi-layered access controls. This includes perimeter security (fences, gates), building access (security guards, key card/badge readers), and data centre floor access (biometric scanners like fingerprint or iris scanners, PIN pads, mantraps). Maintain detailed access logs.
    *   **Prevention/Mitigation:** Prevents unauthorized individuals from physically entering sensitive areas, reducing the risk of theft, sabotage, or tampering.

2.  **Environmental Monitoring and Control Systems:**
    *   **Measure:** Install and maintain robust environmental controls such as fire detection and suppression systems (e.g., smoke detectors, VESDA, gas-based suppression like FM-200 or Novec 1230), temperature and humidity sensors with alerts, and water leakage detectors. Ensure redundant HVAC systems and power supplies (UPS, generators).
    *   **Prevention/Mitigation:** Mitigates damage from environmental threats like fire, overheating, or water damage, protecting hardware and data.

3.  **Surveillance and Intrusion Detection:**
    *   **Measure:** Install comprehensive CCTV (Closed-Circuit Television) coverage for all critical areas, including entrances, exits, server aisles, and perimeter. Implement intrusion detection systems (motion sensors, door/window contacts) that trigger alarms and alert security personnel.
    *   **Prevention/Mitigation:** Deters potential intruders, provides evidence in case of an incident, and allows for rapid response to unauthorized access attempts or suspicious activity.

4.  **Secure Equipment Installation and Management:**
    *   **Measure:** Securely mount servers and network equipment in locked racks or cages. Implement clear desk and clear screen policies. Control and track all media (tapes, USB drives). Implement secure procedures for decommissioning and disposing of old hardware and media (e.g., degaussing, physical destruction).
    *   **Prevention/Mitigation:** Reduces the risk of casual theft of components or media, prevents unauthorized access to data on unattended workstations or exposed media, and ensures data is properly destroyed when no longer needed.

**b) Besides preparing a data redundancy plan, you are also required to prepare an IT security plan for S&W Sdn Bhd.**

**(i) Examine the purpose of an IT security plan. (2 marks)**
The purpose of an IT security plan is to provide a formal, documented strategy outlining how S&W Sdn Bhd will protect its information assets (data, systems, networks) from various threats and vulnerabilities. It aims to ensure the confidentiality, integrity, and availability of these assets, align security efforts with business goals, manage risks effectively, and ensure compliance with relevant regulations and standards.

**(ii) Illustrate THREE (3) important information that an IT security plan should include. (6 marks)**

1.  **Security Policies and Procedures:**
    *   **Information:** This section would detail the overarching security rules and specific step-by-step instructions that employees and systems must follow.
    *   **Illustration:**
        *   **Acceptable Use Policy (AUP):** Defines how employees are permitted to use company IT resources (e.g., internet, email, software).
        *   **Password Policy:** Specifies requirements for creating and managing strong passwords (length, complexity, change frequency).
        *   **Incident Response Procedure:** Outlines the steps to be taken when a security incident is detected (e.g., who to contact, how to contain the incident, how to recover).
        *   **Data Backup and Recovery Policy:** Details how data should be backed up, how often, where backups are stored, and the procedure for restoring data.

2.  **Risk Assessment and Management Strategy:**
    *   **Information:** This section identifies potential threats to S&W Sdn Bhd's IT assets, assesses vulnerabilities, and outlines the strategies to manage these risks.
    *   **Illustration:**
        *   **Asset Inventory:** A list of critical IT assets (e.g., customer database server, financial application, core network routers).
        *   **Threat Identification:** Examples of threats like malware, phishing, DoS attacks, data breaches, natural disasters.
        *   **Vulnerability Assessment Results:** Summary of identified weaknesses in systems or processes.
        *   **Risk Treatment Plan:** How each identified risk will be handled (e.g., "Implement multi-factor authentication to mitigate risk of credential theft," "Purchase cyber insurance to transfer risk of financial loss from certain breaches").

3.  **Roles, Responsibilities, and Awareness Training:**
    *   **Information:** This section clearly defines who is responsible for various aspects of IT security and outlines the plan for security awareness training.
    *   **Illustration:**
        *   **Security Roles:** Defining the responsibilities of the IT Security Officer, system administrators, department heads, and individual users regarding security. For instance, "The IT Manager is responsible for ensuring all servers are patched within 7 days of patch release."
        *   **Security Awareness Program Outline:** Details of mandatory security training for all employees (e.g., annual phishing awareness training, new employee security induction).
        *   **Contact Information:** Key contacts for reporting security incidents or seeking security advice.

---
This completes Question 2 of the January 2024 paper. I will proceed with Question 3 next.

---

**Question 3 (January 2024)**

**a) A Denial of Service (DoS) attack is one of the common cyberattacks nowadays.**

**(i) Illustrate how an attacker uses IP spoofing technique in causing a target computer cannot respond to other legitimate requests. (8 marks)**

IP spoofing in a DoS attack, particularly in a reflective or amplification attack (like a Smurf attack or DNS amplification), works as follows:

**Scenario: SYN Flood with Spoofed IP (Direct DoS)**
While reflective attacks are common with IP spoofing, let's illustrate a direct SYN flood as it's simpler to grasp the core spoofing concept in DoS.

1.  **Target Identification:** The attacker identifies the target computer (Victim Server) they wish to render unresponsive. Let's say its IP address is `V_IP`.

2.  **Spoofing the Source IP Address:** The attacker prepares to send a flood of TCP SYN packets to the Victim Server. Instead of using their own real IP address as the source IP in these packets, the attacker forges (spoofs) the source IP address.
    *   The attacker might use **random, non-existent IP addresses**.
    *   The attacker might use the **IP address of another unsuspecting machine** (which could then become a secondary victim of response packets if the protocol expects a response).

3.  **Sending SYN Packets:** The attacker's machine(s) (often a botnet for a Distributed DoS - DDoS) sends a massive volume of these SYN packets with spoofed source IPs to a specific port (e.g., port 80 for a web server) on the Victim Server (`V_IP`).
    *   A SYN packet is the first step in the TCP three-way handshake (SYN, SYN-ACK, ACK) used to establish a connection.

4.  **Victim Server's Response (Attempting to Complete Handshake):**
    *   When the Victim Server receives a SYN packet, it allocates resources (memory, CPU cycles) to track this potential new connection. It creates an entry in its connection table (often called a "half-open connection").
    *   The Victim Server then sends a SYN-ACK packet back to the *spoofed source IP address* found in the incoming SYN packet.

5.  **Consequences of Spoofing:**
    *   If the spoofed IP address is **non-existent or unreachable**, the Victim Server's SYN-ACK packet goes nowhere. The Victim Server waits for the final ACK packet (which will never arrive from that spoofed IP).
    *   If the spoofed IP address belongs to **another live machine**, that machine will receive an unsolicited SYN-ACK. Since it never sent a SYN, it will typically respond with an RST (reset) packet to the Victim Server, but this doesn't alleviate the Victim Server's primary problem.

6.  **Resource Exhaustion on Victim Server:**
    *   The attacker continues to flood the Victim Server with SYN packets from many different spoofed source IPs.
    *   The Victim Server's connection table fills up with these half-open connections. Each entry consumes resources.
    *   Eventually, the server's resources (memory, connection table capacity) are exhausted. It can no longer accept new incoming connections, including those from legitimate users.
    *   Legitimate users trying to connect will have their SYN packets dropped or ignored, effectively denying them service.

**Illustration Diagram:**

```
Attacker (Spoofed Source IPs: S_IP1, S_IP2, S_IP3...)
    |
    |---- SYN (Src: S_IP1, Dst: V_IP) ----> Victim Server (V_IP)
    |                                        (Allocates resources for S_IP1)
    |---- SYN (Src: S_IP2, Dst: V_IP) ----> (Sends SYN-ACK to S_IP1 - goes nowhere)
    |                                        (Allocates resources for S_IP2)
    |---- SYN (Src: S_IP3, Dst: V_IP) ----> (Sends SYN-ACK to S_IP2 - goes nowhere)
    :                                        (Allocates resources for S_IP3)
    :                                        (Sends SYN-ACK to S_IP3 - goes nowhere)
    (Many more SYN packets)                  :
                                             (Connection table fills up)
                                             (Resources exhausted)

Legitimate User (L_IP)
    |
    |---- SYN (Src: L_IP, Dst: V_IP) -----> X (Victim Server cannot respond/accept)
```

The key is that the victim expends resources trying to respond to non-existent or uninterested spoofed IPs, overwhelming its capacity to handle legitimate requests.

**(ii) Suggest TWO (2) countermeasures for DoS attacks. (6 marks)**

1.  **Ingress and Egress Filtering (Anti-Spoofing):**
    *   **Countermeasure:** Internet Service Providers (ISPs) and network administrators should implement filtering on their network routers.
        *   **Ingress Filtering (at ISP/Network Edge):** Routers at the edge of a network check incoming packets. If a packet arrives on an interface with a source IP address that is not within the range of IP addresses expected from that interface (e.g., a packet arriving from outside the network claiming to have an internal source IP), it is dropped. This helps prevent attackers within a network from launching attacks with spoofed IPs from *outside* their legitimate IP range.
        *   **Egress Filtering (at Source Network):** Networks should filter outgoing traffic to ensure that packets leaving their network have a source IP address that legitimately belongs to their network. This prevents users or compromised machines within their network from launching DoS attacks with spoofed source IPs.
    *   **How it helps:** While it doesn't stop all DoS attacks (especially those that don't rely on IP spoofing or are volumetric), it significantly reduces the effectiveness of IP spoofing-based DoS attacks by making it harder for attackers to hide their true origin or use certain amplification techniques.

2.  **Traffic Scrubbing Centers / DDoS Mitigation Services:**
    *   **Countermeasure:** Organizations can subscribe to specialized DDoS mitigation services. These services have large-capacity networks and sophisticated tools to detect and filter malicious DoS traffic before it reaches the organization's own network.
    *   **How it helps:** When a DoS attack is detected, the organization's internet traffic is rerouted through the mitigation provider's "scrubbing centers." These centers use various techniques (e.g., rate limiting, SYN cookies, behavioral analysis, IP reputation filtering, challenge-response mechanisms) to identify and drop malicious attack traffic while allowing legitimate traffic to pass through to the target systems. This is particularly effective against large-volume (volumetric) DoS attacks.

**b) A brute force attack is a common password attack that is used by threat actors.**

**(i) Illustrate a scenario to show how a brute force attack works. (3 marks)**

**Scenario:**
An attacker wants to gain unauthorized access to an online university portal account belonging to a student named "john.doe". The attacker knows John's username.

1.  **Preparation:** The attacker uses a software tool designed for brute-force attacks.
2.  **Attack Execution:** The tool is configured to target the university portal's login page with the username "john.doe". The tool then systematically tries to guess John's password by attempting every possible combination of characters (e.g., "a", "b", ..., "aa", "ab", ..., "password123", "Password123!", etc.) or by using a dictionary of common passwords.
3.  **Outcome:** The tool sends login requests to the portal with "john.doe" and each guessed password. If a guessed password matches John's actual password, the tool will successfully log in, and the attacker gains access. If the portal has weak or no countermeasures, this process continues until the password is found or the attacker gives up.

**(ii) Suggest ONE (1) countermeasure against brute force attacks. (2 marks)**

**Countermeasure: Account Lockout Policy.**
*   **Explanation:** Implement a policy that temporarily locks an account (e.g., for 15-30 minutes or until an administrator unlocks it) after a certain number of failed login attempts (e.g., 3-5 attempts) from the same IP address or for the same username within a short period. This significantly slows down brute-force attacks, making them impractical as the attacker can only try a few passwords before being locked out.

**(iii) Provide THREE (3) common mistakes that lead to creating bad passwords. (6 marks)**

1.  **Using Easily Guessable Information:**
    *   **Mistake:** Creating passwords based on personal information that is easy for an attacker to find or guess.
    *   **Examples:** Names of family members (children, spouse, pets), birthdates, anniversaries, addresses, phone numbers, favorite sports teams, or simple keyboard patterns (e.g., "qwerty", "asdfgh", "12345678").
    *   **Why it's bad:** Attackers often use this information in targeted dictionary attacks.

2.  **Lack of Complexity (Too Short or Not Using Mixed Character Types):**
    *   **Mistake:** Creating passwords that are too short or only use one type of character (e.g., all lowercase letters, or only numbers).
    *   **Examples:** "password", "pass", "admin", "12345", "football".
    *   **Why it's bad:** Shorter passwords and those with limited character sets drastically reduce the number of possible combinations, making them much easier and faster to crack using brute-force techniques. Modern computing power can crack such passwords very quickly.

3.  **Reusing Passwords Across Multiple Accounts:**
    *   **Mistake:** Using the same password (or very similar variations) for multiple online accounts (e.g., email, banking, social media, work).
    *   **Examples:** Using "MySecretP@ssw0rd!" for both a personal email account and an online banking account.
    *   **Why it's bad:** If one service suffers a data breach and user credentials (including passwords) are leaked, attackers can then use those leaked credentials to try and access other accounts owned by the same user (this is called credential stuffing). A single compromised password can then lead to multiple accounts being compromised.

---
This completes Question 3 of the January 2024 paper. I will proceed with Question 4 next.

---

**Question 4 (January 2024)**

**a) Cryptography algorithms are classified into either an asymmetric encryption or a symmetric encryption.**

**(i) With the aid of a diagram, explain how a symmetric encryption works. (10 marks)**

**Explanation of Symmetric Encryption:**
Symmetric encryption, also known as secret-key cryptography or conventional encryption, is a type of encryption where the **same key** is used for both the encryption of plaintext and the decryption of ciphertext. The sender and the receiver must share this secret key before they can communicate securely. The security of symmetric encryption relies heavily on the secrecy of this shared key; if the key is compromised, any attacker who obtains it can decrypt all messages encrypted with that key.

**Process:**
1.  **Key Sharing:** The sender and receiver must securely agree upon and exchange a secret key. This is a critical step and often a challenge in symmetric cryptography.
2.  **Encryption (by Sender):**
    *   The sender takes the original message (plaintext).
    *   The sender applies a symmetric encryption algorithm (e.g., AES, DES, Blowfish) to the plaintext using the shared secret key.
    *   This process transforms the plaintext into an unreadable format called ciphertext.
3.  **Transmission:** The ciphertext is transmitted over an insecure channel (e.g., the internet) to the receiver.
4.  **Decryption (by Receiver):**
    *   The receiver receives the ciphertext.
    *   The receiver applies the corresponding symmetric decryption algorithm (the inverse of the encryption algorithm) to the ciphertext using the **exact same shared secret key** that was used for encryption.
    *   This process transforms the ciphertext back into the original plaintext.

**Diagram of Symmetric Encryption:**

```
+---------------------+                          +---------------------+
|      Plaintext      |                          |      Plaintext      |
| (Original Message)  |                          | (Original Message)  |
+---------------------+                          +---------------------+
          |                                                ^
          | (Sender Side)                                  | (Receiver Side)
          v                                                |
+---------------------+     SHARED SECRET KEY     +---------------------+
| Symmetric           |<----------------------- | Symmetric           |
| Encryption Algorithm|  +-------------------+  | Decryption Algorithm|
| (e.g., AES)         |  |   Same Secret Key |  | (e.g., AES)         |
+---------------------+  +-------------------+  +---------------------+
          |                  /           \                  ^
          |                 /             \                 |
          |                /               \                |
          v               v                 v               |
+---------------------+                                +---------------------+
|     Ciphertext      |       TRANSMISSION             |     Ciphertext      |
| (Scrambled Message) |-------- OVER -------->         | (Scrambled Message) |
|                     |   INSECURE CHANNEL             |                     |
+---------------------+                                +---------------------+
```

**Key Characteristics shown in the diagram:**
*   **Shared Secret Key:** The same key is used by both the encryption and decryption algorithms. This is highlighted as "SHARED SECRET KEY" and shown connecting to both processes.
*   **Two Processes:** Encryption transforms plaintext to ciphertext; decryption transforms ciphertext back to plaintext.
*   **Algorithms:** Specific algorithms (like AES) are used for these transformations.

**Strengths:**
*   Generally faster and computationally less intensive than asymmetric encryption.
*   Good for encrypting large amounts of data.

**Weaknesses:**
*   Secure key distribution is a major challenge: how to share the secret key securely in the first place.
*   Scalability: In a large network, managing unique shared keys between all pairs of users can become complex (N*(N-1)/2 keys for N users).
*   Does not inherently provide non-repudiation or digital signatures.

**(ii) Provide THREE (3) comparisons between an asymmetric encryption and a symmetric encryption. (6 marks)**

| Feature                 | Symmetric Encryption                                     | Asymmetric Encryption                                       |
| :---------------------- | :------------------------------------------------------- | :---------------------------------------------------------- |
| **Keys Used**           | Uses a **single, shared secret key** for both encryption and decryption. | Uses a **pair of keys**: a public key (for encryption) and a private key (for decryption). The private key is kept secret, while the public key can be widely distributed. |
| **Key Management**      | **Key distribution is a major challenge.** Securely sharing the secret key with the intended recipient without interception is difficult. Poor scalability for large numbers of users due to the number of unique keys required. | **Key distribution is simpler.** The public key can be distributed openly. The private key never needs to be transmitted. Better scalability for key management. |
| **Speed / Performance** | Generally **faster** and computationally less intensive. More suitable for encrypting large volumes of data. | Generally **slower** and more computationally intensive due to complex mathematical operations. Often used to encrypt smaller amounts of data (like symmetric keys or for digital signatures). |
| **Primary Use Cases**   | Bulk data encryption (e.g., encrypting files, full disk encryption, secure communication sessions once a key is established). | Secure key exchange (e.g., establishing a shared secret key for symmetric encryption), digital signatures (for authenticity, integrity, and non-repudiation), and encryption of small amounts of data. |
| **Also Known As**       | Secret-key cryptography, conventional encryption.          | Public-key cryptography.                                    |

**b) Assume that you are asked to design the authentication module for an e-wallet application. Suggest and explain THREE (3) methods of authentication to be considered for the e-wallet application. (9 marks)**

For an e-wallet application, robust authentication is critical due to the direct financial implications. A multi-layered approach, often involving Multi-Factor Authentication (MFA), is best. Here are three methods:

1.  **Password/PIN (Something You Know):**
    *   **Suggestion:** A strong password or a dedicated PIN (Personal Identification Number) for accessing the e-wallet application itself.
    *   **Explanation:** This is a fundamental first layer of authentication.
        *   **Password:** Should enforce complexity rules (length, mix of character types) and be unique to the e-wallet.
        *   **PIN:** Often a shorter numeric code (e.g., 6 digits) used for quick access once the app is opened or for authorizing transactions. This is typically device-specific.
    *   **Security Consideration:** While essential, passwords/PINs alone are vulnerable to phishing, shoulder surfing, keylogging, or brute-force attacks if not protected by other measures. It's important to implement account lockout policies and encourage users to choose strong, unique credentials.

2.  **Biometric Authentication (Something You Are):**
    *   **Suggestion:** Integrate biometric authentication methods available on the user's smartphone, such as fingerprint scanning or facial recognition.
    *   **Explanation:** This method uses unique biological characteristics of the user for verification.
        *   **Fingerprint Scan:** User places their finger on the phone's sensor.
        *   **Facial Recognition:** User looks at the phone's camera.
    *   **Security Consideration:** Biometrics are convenient and generally harder to spoof than passwords (though not impossible). They tie authentication to the physical user and their specific device. This significantly enhances security, especially against remote attacks trying to use stolen credentials, as the attacker would also need physical access to the user or their enrolled biometric data. It's often used as a quick way to unlock the app or authorize payments after initial setup.

3.  **One-Time Password (OTP) via Authenticator App or SMS (Something You Have - for specific high-risk actions):**
    *   **Suggestion:** Implement OTPs for critical actions like logging in from a new device, resetting a password, adding a new beneficiary, or authorizing large transactions.
    *   **Explanation:** An OTP is a temporary code, valid for only one login session or transaction.
        *   **Authenticator App (e.g., Google Authenticator, Authy):** The e-wallet app is linked to an authenticator app on the user's phone. The authenticator app generates time-based OTPs (TOTP) that change every 30-60 seconds. This is generally more secure than SMS OTPs.
        *   **SMS OTP:** A code is sent to the user's registered mobile number via SMS.
    *   **Security Consideration:** This provides a strong second factor of authentication. Even if an attacker has the user's password (Factor 1: Something You Know), they would also need access to the user's phone to receive the OTP (Factor 2: Something You Have) to complete the authentication or authorize the transaction. This significantly mitigates risks from phishing and credential theft. Authenticator apps are preferred over SMS due to the risk of SIM swapping attacks with SMS OTPs.

**Combining these methods for Multi-Factor Authentication (MFA) is the best approach for an e-wallet:**
*   **Login:** Password/PIN + Biometrics (if available/enabled)
*   **High-Risk Transactions:** May require an additional OTP or re-authentication with PIN/Biometrics.

---
This completes the January 2024 paper. Let me know which one is next. I see "May/June 2024" and "May/June 2023". Let's do **May/June 2023 (the last image set from the first interaction).**

---

**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**
**FACULTY OF COMPUTING AND INFORMATION TECHNOLOGY**
# **ACADEMIC YEAR 2022/2023**
**MAY/JUNE EXAMINATION**
**INFORMATION TECHNOLOGY BAIT1093**
**INTRODUCTION TO COMPUTER SECURITY**

---

**Question 1**

**a) Briefly describe a file-based virus. Identify TWO (2) malicious actions that a file-based virus can do to a host machine. (4 marks)**

**Description of a file-based virus:**
A file-based virus (or file infector virus) is a type of malware that attaches its malicious code to legitimate executable files (e.g., .EXE, .COM, .DLL files) or sometimes to data files that can contain executable macros (e.g., Microsoft Office documents). When the infected file is executed or opened, the virus code also runs, allowing it to replicate by infecting other files and carry out its malicious payload.

**TWO (2) malicious actions:**
1.  **Data Deletion or Corruption:** The virus can be programmed to delete important system files or user data, or to corrupt files by overwriting them with garbage data, rendering them unusable and potentially making the system unstable or unbootable.
2.  **System Slowdown or Crashing:** By consuming system resources (CPU, memory, disk space) for its replication or payload activities, the virus can significantly slow down the host machine or cause frequent system crashes and instability.

**b) Your friend, Melanie, found a USB flash drive at her university's computer lab. Out of curiosity, she plugged the USB flash drive into her laptop. You did not manage to stop her on time and shared with her that the USB flash drive may contain malicious programs like a keylogger. Explain TWO (2) malicious actions that a keylogger can do. (4 marks)**

A keylogger is a type of spyware that records every keystroke made on a computer's keyboard.

**TWO (2) malicious actions of a keylogger:**
1.  **Steal Login Credentials:** The keylogger can capture usernames and passwords as Melanie types them into websites (e.g., online banking, email, social media) or applications. This allows the attacker who deployed the keylogger to gain unauthorized access to her accounts.
2.  **Capture Sensitive Personal Information:** It can record other sensitive information Melanie types, such as credit card numbers, bank account details (when filling online forms), private messages, confidential documents, or search queries. This information can be used for identity theft, financial fraud, or extortion.

**c) "Zero trust principles and technologies can reduce or eliminate many attack vectors and provide valuable data for ongoing attack surface analysis."**
**Source: https://www.ibm.com/my-en/topics/attack-surface**

**(i) What is an attack vector? (2 marks)**
An attack vector is a specific path, method, or technique used by a threat actor to gain unauthorized access to a system, application, or network to deliver a malicious payload, exploit a vulnerability, or achieve a malicious objective. It's the "how" an attacker can breach security.

**(ii) Explain TWO (2) attack vectors that may be used by a threat actor. (10 marks)**
*(This is similar to Jan 2024 Q1a. I'll provide slightly different examples or emphasis.)*

1.  **Compromised Credentials:**
    *   **Explanation:** This vector involves an attacker obtaining and using legitimate user credentials (username and password, API keys, access tokens) to gain unauthorized access. Credentials can be compromised through various means such as phishing, data breaches of third-party services where users reuse passwords, malware (like keyloggers or info-stealers), or brute-force/dictionary attacks against weak passwords.
    *   **How it's used:** Once an attacker has valid credentials, they can log into systems, applications, or networks impersonating a legitimate user. This bypasses many perimeter defenses as the system sees the login as authentic. The attacker can then access sensitive data, install further malware, escalate privileges, or move laterally within the network.
    *   **Example:** An attacker obtains an employee's VPN credentials through a phishing attack. They then use these credentials to connect to the company's internal network, gaining access to internal resources.

2.  **Insider Threats (Malicious or Negligent):**
    *   **Explanation:** This vector originates from individuals who have legitimate, authorized access to an organization's systems and data, such as current or former employees, contractors, or business partners.
        *   **Malicious Insider:** Intentionally misuses their access to steal data, sabotage systems, or commit fraud, often driven by financial gain, revenge, or ideology.
        *   **Negligent Insider:** Unintentionally causes a security incident through carelessness, errors, or failure to follow security policies (e.g., clicking on a phishing link, misconfiguring a server, losing a company laptop).
    *   **How it's used:** A malicious insider might directly copy sensitive files to a USB drive, email confidential data to an external account, or plant logic bombs. A negligent insider might accidentally expose data on an unsecured cloud storage or fall victim to social engineering.
    *   **Example:** A disgruntled employee with database administrator privileges copies the entire customer database before leaving the company and sells it on the dark web.

**(iii) How can zero trust principles reduce or eliminate many attack vectors? (5 marks)**

Zero Trust is a security model based on the principle of "never trust, always verify." It assumes that threats can originate from both outside and inside the network, so no user or device should be trusted by default, regardless of its location.

Zero Trust principles can reduce or eliminate attack vectors in several ways:
1.  **Microsegmentation and Granular Access Control:** Zero Trust advocates for dividing the network into small, isolated segments (microsegments). Access policies are strictly enforced at a very granular level (per-application, per-user, per-device) based on verified identity, device security posture, and context. This drastically reduces the "blast radius" if one segment is compromised, as an attacker cannot easily move laterally (eliminating or reducing the effectiveness of lateral movement as an attack vector).
2.  **Strict Identity Verification (Multi-Factor Authentication - MFA):** Every user and device attempting to access any resource must be rigorously authenticated and authorized, typically using strong MFA. This significantly reduces the effectiveness of compromised credentials as an attack vector because a password alone is insufficient.
3.  **Principle of Least Privilege:** Users and applications are granted only the minimum necessary permissions required to perform their specific tasks. This limits what an attacker can do even if they compromise an account or system, reducing the impact of vectors like malware or insider threats.
4.  **Continuous Monitoring and Validation:** Zero Trust environments continuously monitor user behavior, device health, and network traffic for anomalies. Trust is not static; it's re-evaluated dynamically. If suspicious activity is detected, access can be automatically revoked or restricted, quickly mitigating an active attack vector.
5.  **Assuming Breach:** By assuming that attackers are already present or will inevitably get in, Zero Trust focuses on containing them and minimizing their impact, rather than solely relying on perimeter defenses. This makes many traditional attack vectors that rely on a trusted internal network less effective.

For example, if an attacker uses phishing to steal credentials (attack vector), Zero Trust's MFA requirement would prevent them from logging in. If malware lands on a device (attack vector), microsegmentation and least privilege would limit its ability to spread or access critical data.

---
I'll proceed with Question 2 of the May/June 2023 paper next.Okay, continuing with the **May/June 2023 Examination**.

---

**Question 2**

**Assume you are the IT Security Manager of a company. The company office is located on one floor of a building. Within the office space, there is an office lobby, four rooms where two rooms (Room 3 and Room 4) are protected and one confidential room where the computer facility is located. Room 1 and Room 2 can be used by anyone, either by staff or visitors that have been given clearance by a receptionist stationed outside the lobby. Room 3 and Room 4 are the rooms for staff and senior management respectively. Computer Facility is referring to server rooms, hosting applications and databases as well as network equipment.**

**(Figure 2.1 shows a layout: Lobby leads to A. A is a corridor. Off A are Room 1 & Room 2 (Public). Also off A is a door leading to another corridor B. Off B are Room 3 & Room 4 (Protected). Off B is also another door leading to corridor C. Corridor C leads to the Computer Facility (Confidential).)**

**a) Based on Figure 2.1, suggest suitable physical security measures to be in place at locations A, B and C to prevent data or hardware loss/theft from the company to access public, protected and confidential areas respectively. Explain your suggested physical security measures for each area. You can suggest one or more physical security measures at each location suitable for protecting company's data or hardware at each location. Use the following table format on your answer script to answer the question. (18 marks)**

| Location | Suggested Physical Security Measures                                                                                                                               | Explanation                                                                                                                                                                                                                                                                                                                                                        |
| :------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** (Corridor leading from Lobby to Public Rooms 1&2 and to entrance of Area B) | 1. **CCTV Surveillance:** Visible and covert cameras covering the entire corridor. <br> 2. **Access Point to Area B:** A locked door requiring authentication (e.g., key card reader) to enter Area B. <br> 3. **Clear Signage:** Indicating "Public Access Area" and restricted access beyond a certain point (leading to B). <br> 4. **Security Personnel/Receptionist Visibility:** Ensure the receptionist at the lobby has a clear line of sight or CCTV feed of Area A if possible. | **CCTV:** Deters unauthorized activity and provides evidence if an incident occurs. Acts as a general surveillance for the public access point. <br> **Access Point to B:** This is the first layer of control separating public areas from protected areas. It ensures only authorized staff can proceed further. <br> **Signage:** Manages expectations and informs visitors of access limitations, deterring casual attempts to enter restricted zones. <br> **Receptionist:** Can monitor activity and challenge suspicious individuals. |
| **B** (Corridor leading to Protected Rooms 3&4 and to entrance of Area C) | 1. **Access Control for Entry to B:** Already mentioned above (key card reader at the door from A to B). <br> 2. **CCTV Surveillance:** Full coverage of corridor B. <br> 3. **Stronger Access Control for Entry to Area C:** A more robust locked door (e.g., steel door) with multi-factor authentication (e.g., key card + PIN, or biometric scanner) to enter Area C (leading to Computer Facility). <br> 4. **Intrusion Detection Sensors:** Motion detectors in corridor B, especially active after hours. | **Access Control to B:** Ensures only staff can enter this semi-restricted zone. <br> **CCTV:** Monitors staff movement and any unauthorized attempts to access Rooms 3/4 or Area C. <br> **Stronger Access Control to C:** This is a critical control point before the most sensitive area. MFA significantly increases security. <br> **Intrusion Detection:** Alerts security if unauthorized presence is detected, especially during off-peak hours. |
| **C** (Corridor/Airlock leading to Computer Facility - Confidential) | 1. **High-Security Door with Multi-Factor Authentication:** As mentioned (e.g., biometric + card). Potentially a mantrap system (two interlocking doors where only one can be open at a time). <br> 2. **CCTV Surveillance:** Focused on the entrance to the Computer Facility. <br> 3. **Access Logging:** All access attempts (successful and failed) to the Computer Facility door must be logged and audited regularly. <br> 4. **No Unescorted Visitor Access:** Strict policy that no visitors are allowed in Area C or the Computer Facility without authorized escort and prior approval. <br> 5. **Physical Intrusion Detection on Walls/Ceiling/Floor:** For the Computer Facility itself, consider sensors that detect attempts to breach physical barriers. | **High-Security Door/Mantrap:** Provides the highest level of access control to the most critical asset. Mantraps prevent tailgating. <br> **CCTV:** Records everyone entering/exiting the most sensitive area. <br> **Access Logging:** Crucial for auditing and investigating any security incidents, and for accountability. <br> **Escort Policy:** Prevents unauthorized individuals from gaining access even if accompanying staff. <br> **Intrusion Detection (Structural):** Protects against forced entry through non-door pathways. |

**b) Among managerial controls, operational controls and technical controls, which security controls that you are using to implement the suggested physical security measures based on your answer from Question 2 a)? Explain your answer. (7 marks)**

The suggested physical security measures from Question 2 a) utilize a combination of all three types of security controls:

1.  **Technical Controls:**
    *   **Usage:** These are controls implemented using technology.
    *   **Examples from 2a):**
        *   **CCTV Surveillance systems:** Technology for recording and monitoring.
        *   **Electronic Access Control Systems:** Key card readers, PIN pads, biometric scanners.
        *   **Intrusion Detection Sensors:** Motion detectors, door/window contact sensors.
        *   **Mantraps:** A physical-technical system for controlling entry.
        *   **Access Logging Systems:** Automated systems that record access attempts.
    *   **Explanation:** These technologies directly enforce access restrictions, monitor areas, and detect unauthorized activities.

2.  **Operational Controls:**
    *   **Usage:** These are controls implemented primarily by people through procedures and day-to-day activities. They support and manage the technical controls.
    *   **Examples from 2a):**
        *   **Security Personnel/Receptionist Monitoring:** Human oversight and intervention (e.g., receptionist clearing visitors, security guards patrolling or responding to alarms).
        *   **Access Log Auditing:** People reviewing the logs generated by technical controls.
        *   **Escort Policy for Visitors:** A procedure that staff must follow.
        *   **Maintenance of Security Systems:** Regular checks and upkeep of CCTV, access control systems, etc., performed by personnel.
    *   **Explanation:** These involve human actions and established procedures to ensure security measures are effective and followed, such as monitoring CCTV feeds, escorting visitors, and regularly checking that locks are functional.

3.  **Managerial Controls:**
    *   **Usage:** These are high-level controls that provide guidance, policy, and oversight for the security program. They are often administrative in nature.
    *   **Examples from 2a) (often implied or underpinning the other controls):**
        *   **Security Policies:** The underlying company policies that mandate the use of access controls, visitor management, and define areas as public, protected, or confidential (e.g., "No Unescorted Visitor Access policy").
        *   **Risk Assessment:** The initial process that would have identified the need for these varying levels of security at A, B, and C.
        *   **Assignment of Responsibilities:** Defining who is responsible for managing and enforcing physical security (e.g., the IT Security Manager, facilities manager).
        *   **Security Awareness Training:** Training staff on the importance of physical security and their role in maintaining it (e.g., not propping doors open, challenging strangers).
        *   **Clear Signage:** This is a directive, often stemming from a policy, to inform and guide behavior.
    *   **Explanation:** These controls set the direction for security. For example, the decision to classify the Computer Facility as "Confidential" and require MFA is a managerial decision based on risk assessment, which then leads to the implementation of technical and operational controls.

In summary, effective physical security, as suggested for locations A, B, and C, relies on the integrated use of technical (the tools), operational (how the tools and processes are used by people), and managerial (the overarching strategy, policies, and risk decisions) controls.

---
I'll proceed with Question 3 of the May/June 2023 paper next.

---

**Question 3 (May/June 2023)**

**The adoption of cloud computing is very common nowadays in the industry. Although it provides convenience to companies to host applications and for storage purpose, however, there is a lot of security aspects that need to be closely monitored so that data does not fall into wrong hands. Encryption is implemented in cloud computing. Symmetric and asymmetric encryption are the two main types of encryption used in cloud computing. In addition, the right access control models have to be used to maintain full security using cloud computing.**

**a) Show how symmetric encryption differs from asymmetric encryption with the aid of diagrams, assuming the sender's name is Bob and the receiver's name is Alice. Briefly explain ONE (1) main difference between symmetric and asymmetric encryption. (15 marks)**

*(This question is very similar to Jan 2024 Q4a, but with specific names and a table format for marks. I will tailor the diagrams and explanation.)*

**ONE (1) main difference:**
The main difference is that **symmetric encryption uses a single, shared secret key for both encryption and decryption**, while **asymmetric encryption uses a pair of keys: a public key for encryption and a corresponding private key for decryption.**

---
**Diagram to illustrate symmetric encryption (Bob sending to Alice):**

```
                       SHARED SECRET KEY
                      +-------------------+
                      |   Secret Key K    |
                      +-------------------+
                             /     \
                            /       \
                           v         v
Bob (Sender)                                     Alice (Receiver)
+-----------+                                    +-------------+
| Plaintext |                                    |  Plaintext  |
+-----------+                                    +-------------+
      |                                                  ^
      |                                                  |
      v                                                  |
+---------------------+                             +----------------------+
| Symmetric           |---- (Uses Secret Key K) -->| Symmetric            |
| Encryption Algorithm|                             | Decryption Algorithm |
| (e.g., AES)         |---- (Uses Secret Key K) -->| (e.g., AES)          |
+---------------------+                             +----------------------+
      |                                                  ^
      |                                                  |
      v                TRANSMISSION                    |
+------------+         OVER INSECURE        +------------+
| Ciphertext |--------    CHANNEL   -------->| Ciphertext |
+------------+                              +------------+
```
**Explanation of Symmetric Diagram:**
Bob wants to send a plaintext message to Alice. Both Bob and Alice possess the *same secret key K*. Bob encrypts the plaintext using an encryption algorithm (like AES) with the secret key K to produce ciphertext. This ciphertext is sent to Alice. Alice then uses the *same secret key K* and the corresponding decryption algorithm to convert the ciphertext back into the original plaintext.

---
**Diagram to illustrate asymmetric encryption (Bob sending to Alice):**

```
Alice's Key Pair:
+-------------------+   (Kept Secret by Alice)
| Alice's Private Key |
+-------------------+
      ^
      | (Used for Decryption)
      |
+-------------------+   (Shared with Bob / Publicly Available)
| Alice's Public Key  |
+-------------------+
      |
      | (Used for Encryption by Bob)
      v
Bob (Sender)                                     Alice (Receiver)
+-----------+                                    +-------------+
| Plaintext |                                    |  Plaintext  |
+-----------+                                    +-------------+
      |                                                  ^
      |                                                  |
      v                                                  |
+---------------------+                             +----------------------+
| Asymmetric          |<----(Uses Alice's Public Key)-| Asymmetric           |
| Encryption Algorithm|                             | Decryption Algorithm |
| (e.g., RSA)         |----(Uses Alice's Private Key)->| (e.g., RSA)          |
+---------------------+                             +----------------------+
      |                                                  ^
      |                                                  |
      v                TRANSMISSION                    |
+------------+         OVER INSECURE        +------------+
| Ciphertext |--------    CHANNEL   -------->| Ciphertext |
+------------+                              +------------+
```
**Explanation of Asymmetric Diagram:**
Alice generates a key pair: a public key and a private key. She keeps her private key secret and makes her public key available to Bob (and others). Bob wants to send a plaintext message to Alice. Bob uses *Alice's public key* and an asymmetric encryption algorithm (like RSA) to encrypt the plaintext, producing ciphertext. This ciphertext is sent to Alice. Alice then uses *her own private key* and the corresponding decryption algorithm to convert the ciphertext back into the original plaintext. Only Alice, with her private key, can decrypt messages encrypted with her public key.

---
**Assessment Criteria Table:**

| Assessment Criteria                             | Allocation of Marks |
| :---------------------------------------------- | :------------------ |
| Diagram to illustrate asymmetric encryption     | 5 marks             |
| Diagram to illustrate symmetric encryption      | 5 marks             |
| Explanation of ONE (1) main difference        | 5 marks             |

*(The explanation of the main difference is provided above the diagrams).*

**b) If you have the choice to choose an access control model for data stored on a cloud, would you choose a Role-Based Access Control (RBAC) or an Attribute-Based Access Control (ABAC)? Briefly explain the reason for your choice. (5 marks)**

For data stored on a cloud, especially in complex and dynamic environments, I would lean towards **Attribute-Based Access Control (ABAC)**, or at least a hybrid model where ABAC complements RBAC.

**Reason for choosing ABAC (or a hybrid incorporating it):**
Cloud environments often involve diverse users, numerous resources, varying contexts of access, and a need for fine-grained control that goes beyond static roles.
1.  **Granularity and Dynamic Policies:** ABAC allows for highly granular access decisions based on a rich set of attributes related to the user (e.g., role, department, security clearance, location), the resource (e.g., data sensitivity, creation date, project tag), the action being performed (e.g., read, write, delete), and the environment (e.g., time of day, IP address, device security posture). This dynamic nature is well-suited for the cloud's flexibility. For example, a policy could state: "Allow 'Researchers' (user attribute) to 'Read' (action attribute) 'Project X Data' (resource attribute) only if accessing from a 'University IP Address' (environment attribute) during 'Business Hours' (environment attribute)."
2.  **Scalability for Complex Scenarios:** As the number of users, resources, and access rules grows in a cloud environment, managing this with RBAC alone can lead to "role explosion" (creating too many specific roles). ABAC can handle this complexity more efficiently by defining policies based on attributes rather than creating numerous distinct roles for every combination of permissions.
3.  **Context-Aware Security:** ABAC inherently supports context-aware security, which is crucial in the cloud where users might access data from various devices and locations. Access can be granted or denied based on real-time contextual information, enhancing security.
4.  **Reduced Need for Role Re-definition:** When business needs change, policies can often be updated by modifying attribute rules rather than restructuring the entire role hierarchy, making it more adaptable.

While RBAC is simpler for straightforward scenarios with clearly defined job functions, the dynamic and often complex nature of cloud data access benefits significantly from the flexibility and fine-grained control offered by ABAC. A common approach is to use RBAC for broad entitlements and then layer ABAC on top for more specific, context-dependent access decisions.

**c) With encryption and access control models incorporated into cloud computing, do you find that there is a need for organisations to produce risk registers? Explain your answer. (5 marks)**

**Yes, there is absolutely still a need for organisations to produce and maintain risk registers**, even with robust encryption and access control models in cloud computing.

**Explanation:**
1.  **No System is Perfectly Secure:** Encryption and access controls are critical security measures, but they are not infallible.
    *   Encryption keys can be mismanaged, stolen, or compromised.
    *   Access control policies can be misconfigured or have loopholes.
    *   Vulnerabilities can exist in the encryption algorithms or access control implementations themselves.
    *   Insider threats (malicious or negligent) can bypass or misuse legitimate access.
2.  **Risks Beyond Technical Controls:** Risk registers address a broader spectrum of risks than just those mitigated by encryption and access controls. These include:
    *   **Cloud Provider Risks:** Risks associated with the cloud service provider (CSP) itself, such as CSP outages, CSP security breaches, changes in CSP terms of service, or the CSP going out of business (vendor lock-in/viability).
    *   **Compliance and Legal Risks:** Ensuring data residency requirements, meeting industry-specific regulations (e.g., HIPAA, PCI DSS), and managing legal liabilities related to data in the cloud.
    *   **Operational Risks:** Misconfigurations by internal staff, lack of skilled personnel to manage cloud security, data loss due to human error despite backups.
    *   **Physical Security Risks (at CSP):** While the CSP is responsible, the organization still bears some residual risk if the CSP's physical security fails.
    *   **Business Continuity and Disaster Recovery Risks:** Ensuring that data can be recovered and services restored within acceptable timeframes if the cloud service is disrupted.
3.  **Risk Management is an Ongoing Process:** A risk register is a key tool in the continuous risk management lifecycle. It helps organizations to:
    *   **Identify** new and emerging threats relevant to their cloud deployment.
    *   **Assess** the likelihood and impact of these threats.
    *   **Prioritize** risks for treatment.
    *   **Track** the implementation and effectiveness of mitigation measures (which might include enhancing encryption or access controls).
    *   **Communicate** risks to stakeholders.
4.  **Informing Security Strategy:** The risk register helps justify security investments, including those in encryption and access control, by quantifying the risks they mitigate. It also helps identify gaps where existing controls may be insufficient.

In conclusion, while encryption and access controls are vital components of cloud security, they are just parts of an overall risk management strategy. A risk register remains essential for a holistic approach to identifying, assessing, and managing the diverse range of risks associated with using cloud computing.

---
I'll proceed with Question 4 of the May/June 2023 paper next.

---

**Question 4 (May/June 2023)**

**Ransomware attacks have been rising and expect to increase exponentially in the coming years.**

**a) If you look at the aspect of a threat actor from a cybercriminal group, what is the purpose of using ransomware during cyberattacks? Explain your answer. (5 marks)**

From the perspective of a cybercriminal group (threat actor), the primary purpose of using ransomware during cyberattacks is **financial extortion and monetization of the attack.**

**Explanation:**
1.  **Direct Financial Gain (Ransom Payment):** Ransomware directly demands a payment (the ransom) from the victim in exchange for decrypting their files or restoring access to their systems. This provides a clear and relatively straightforward way for cybercriminals to make money from their malicious activities. Cryptocurrencies are often demanded because they offer a degree of anonymity and are harder to trace than traditional financial transactions.
2.  **Leverage for Extortion (Data Exfiltration - Double/Triple Extortion):** Modern ransomware attacks often involve "double extortion" or even "triple extortion."
    *   **Double Extortion:** Before encrypting files, cybercriminals exfiltrate (steal) sensitive data from the victim's network. They then threaten to publish or sell this stolen data if the ransom is not paid, even if the victim can restore their files from backups. This adds immense pressure on the victim to pay.
    *   **Triple Extortion:** This can extend to threatening DDoS attacks against the victim or contacting the victim's customers/partners about the breach if the ransom isn't paid.
    This increases the likelihood of payment by targeting not just data availability but also data confidentiality and the victim's reputation.
3.  **Scalability and Efficiency:** Ransomware-as-a-Service (RaaS) models have emerged, where ransomware developers lease their malware to affiliates who then carry out the attacks. This allows less skilled criminals to participate and scales the overall ransomware operation, increasing profits for the core group. The attacks can be automated and deployed widely.
4.  **Impact and Urgency:** Ransomware causes immediate and significant disruption to a victim's operations, creating a sense of urgency and panic. This often compels victims to pay quickly to restore business continuity, especially if they lack robust backups or an effective incident response plan.

In essence, for cybercriminal groups, ransomware is a highly effective tool for coercing victims into paying large sums of money by holding their critical data and systems hostage and threatening further damaging disclosures.

**b) Which type of firewall will have a higher chance to detect a ransomware attack? Is it a stateless packet inspection firewall or a stateful inspection firewall? Provide your explanation. (5 marks)**

Neither a traditional stateless packet inspection firewall nor a traditional stateful inspection firewall is specifically designed or highly effective at *directly detecting ransomware infection itself* (i.e., the malware executing on an endpoint or encrypting files). However, one might offer slightly better capabilities in detecting *network activities associated* with some ransomware attacks.

Between the two, a **stateful inspection firewall** would have a slightly higher chance, but this chance is still limited for modern ransomware.

**Explanation:**
*   **Stateless Packet Inspection Firewall:**
    *   Operates at the network layer (Layer 3) and sometimes transport layer (Layer 4).
    *   Examines each packet individually and makes filtering decisions based on information in the packet header (source/destination IP, source/destination port, protocol type).
    *   It does not maintain any information about the state of connections (e.g., whether a packet is part of an established session).
    *   **Detection Capability for Ransomware:** Very limited. It cannot inspect packet payloads for malicious code or understand the context of traffic. It might block traffic to known malicious IPs if it has a static blacklist, but this is not specific to ransomware detection.

*   **Stateful Inspection Firewall:**
    *   Also operates at Layers 3 and 4 but maintains a "state table" that tracks active network connections (e.g., TCP sessions).
    *   It makes filtering decisions based on the packet header information *and* the context of the connection derived from the state table. For example, it can ensure that incoming packets belong to an already established session that was initiated from inside the network.
    *   **Detection Capability for Ransomware:**
        *   **Slightly Better (but still limited for ransomware itself):** Because it understands connection states, it can be more effective at preventing unauthorized inbound connections or recognizing anomalous traffic patterns that deviate from normal established sessions.
        *   Some ransomware might attempt to communicate with Command and Control (C2) servers over unusual ports or protocols, or initiate unexpected outbound connections. If a stateful firewall has very granular rules about what outbound connections are permitted from internal hosts, it *might* block some of this C2 traffic, indirectly hindering the ransomware's operation (e.g., preventing it from retrieving an encryption key or exfiltrating data).
        *   However, many ransomware variants use common ports (like HTTP/80 or HTTPS/443) for C2 communication, which stateful firewalls typically allow. They also don't inspect encrypted payloads.

**Conclusion:**
While a stateful firewall is generally more secure than a stateless one, neither is a primary tool for *detecting the ransomware malware itself*. For that, you need Endpoint Detection and Response (EDR), antivirus/anti-malware software, and Next-Generation Firewalls (NGFWs) or Intrusion Prevention Systems (IPS) that perform deep packet inspection and behavioral analysis.

If forced to choose between only stateless and stateful for *any* chance related to network activity, **stateful** is marginally better due to its ability to track connections and potentially spot more anomalous network behavior associated with some stages of a ransomware attack (like C2 communication if it's on unexpected ports or initiated improperly).

**c) Besides deploying firewalls, suggest and explain THREE (3) countermeasures that an organisation can implement to protect against ransomware attacks. (15 marks)**

1.  **Regular and Tested Data Backups (3-2-1 Rule):**
    *   **Suggestion:** Implement a robust backup strategy, often following the 3-2-1 rule:
        *   **3 Copies:** Keep at least three copies of your important data.
        *   **2 Different Media:** Store the copies on at least two different types of storage media (e.g., local disk, network attached storage (NAS), tapes, cloud storage).
        *   **1 Offsite Copy:** Keep at least one copy of the data offsite (e.g., a separate physical location, or in a geographically distinct cloud region). One of these copies should also be **offline (air-gapped)** or **immutable** (cannot be altered or deleted by ransomware).
    *   **Explanation:** This is the most critical countermeasure for *recovery* from a ransomware attack. If an organization's primary data is encrypted, having clean, recent backups allows them to restore their systems and data without paying the ransom. The offline/immutable copy is crucial because ransomware often tries to find and encrypt or delete connected backups. Regularly testing the backup and restoration process is essential to ensure the backups are viable and the recovery procedure works as expected.

2.  **Security Awareness Training and Phishing Simulation:**
    *   **Suggestion:** Conduct regular, ongoing security awareness training for all employees. This training should cover how to identify and report phishing emails, malicious websites, and other social engineering tactics. Supplement this with regular phishing simulation exercises.
    *   **Explanation:** Humans are often the initial entry point for ransomware (e.g., by clicking a malicious link or opening an infected attachment). Training helps employees become more vigilant and less likely to fall for these tricks. Phishing simulations test their understanding and provide practical experience in a safe environment. A well-informed workforce acts as a strong human firewall, significantly reducing the likelihood of initial infection. Training should also emphasize strong password hygiene and safe browsing habits.

3.  **Endpoint Detection and Response (EDR) and Advanced Anti-Malware Solutions:**
    *   **Suggestion:** Deploy advanced endpoint security solutions, such as EDR tools and next-generation antivirus/anti-malware software, on all workstations and servers.
    *   **Explanation:**
        *   **Advanced Anti-Malware:** These tools go beyond traditional signature-based detection. They use heuristics, behavioral analysis, machine learning, and sandboxing to detect and block known and unknown (zero-day) malware, including ransomware variants, often before they can execute or as they attempt to perform malicious actions like file encryption.
        *   **EDR Solutions:** EDR tools provide continuous monitoring and collection of endpoint data (processes, network connections, file access). They can detect suspicious activities indicative of a ransomware attack in progress (e.g., rapid file encryption, attempts to disable security software, unusual network communications). EDR also provides visibility and tools for incident response, allowing security teams to investigate, isolate infected endpoints, and remediate threats more effectively.

---

This completes the May/June 2023 paper. The only one remaining from the images is the **May/June 2024** paper (the fourth image set from the first interaction). Let me know if you want to proceed with that one.

Okay, let's proceed with the **May/June 2024 Examination (the fourth image set from the first interaction).**

---

**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**
**FACULTY OF COMPUTING AND INFORMATION TECHNOLOGY**
# **ACADEMIC YEAR 2023/2024**
**MAY/JUNE EXAMINATION**
**BAIT1093 INTRODUCTION TO COMPUTER SECURITY**

---

**Question 1**

**a) Identify FOUR (4) types of security attacks that modern computer systems may encounter. (4 marks)**

1.  **Malware Attacks:** Infections by malicious software such as viruses, worms, Trojans, ransomware, spyware, adware.
2.  **Phishing Attacks:** Deceptive attempts to acquire sensitive information like usernames, passwords, and credit card details by masquerading as a trustworthy entity in an electronic communication.
3.  **Denial-of-Service (DoS) / Distributed Denial-of-Service (DDoS) Attacks:** Attempts to make a machine or network resource unavailable to its intended users by overwhelming it with a flood of internet traffic or requests.
4.  **Man-in-the-Middle (MitM) Attacks:** Attackers secretly intercept and possibly alter communications between two parties who believe they are directly communicating with each other.

**b) List and explain the SIX (6) Principles of Computer Security. (12 marks)**

While different sources might list slightly varied "principles," common foundational principles often include aspects related to the CIA triad, defense in depth, least privilege, etc. Here are six commonly recognized principles:

1.  **Confidentiality:**
    *   **Explanation:** Ensuring that information is not disclosed to unauthorized individuals, entities, or processes. Data should only be accessible by those who have the proper authorization.
    *   **Relevance:** Protects sensitive data from being stolen or viewed by attackers or unauthorized insiders.

2.  **Integrity:**
    *   **Explanation:** Maintaining the accuracy, consistency, and trustworthiness of data over its entire lifecycle. Data must not be improperly modified, and systems must perform their intended functions in an unimpaired manner.
    *   **Relevance:** Ensures that data is reliable and has not been tampered with, which is critical for decision-making and operational processes.

3.  **Availability:**
    *   **Explanation:** Ensuring that systems, applications, and data are accessible and usable by authorized users when needed. Systems must be resilient and protected against disruptions.
    *   **Relevance:** Guarantees that critical services and information are available for business operations, preventing downtime and associated losses.

4.  **Principle of Least Privilege:**
    *   **Explanation:** Granting users, programs, or processes only the minimum necessary access rights and permissions required to perform their specific tasks or functions, and only for the duration needed.
    *   **Relevance:** Limits the potential damage if an account or system is compromised, as the attacker will have restricted capabilities.

5.  **Defense in Depth (Layered Security):**
    *   **Explanation:** Implementing multiple layers of security controls (physical, technical, and administrative) so that if one layer fails, another layer is in place to protect assets. No single control is relied upon exclusively.
    *   **Relevance:** Increases the difficulty for attackers to succeed, as they must bypass multiple security measures. It provides resilience against the failure of individual controls.

6.  **Separation of Duties:**
    *   **Explanation:** Dividing critical tasks or responsibilities among multiple individuals so that no single person has complete control over a process or can perform all actions related to a sensitive operation. This requires collusion to commit fraud or malicious acts.
    *   **Relevance:** Prevents fraud and errors by ensuring that critical actions are subject to checks and balances involving more than one person.

*(Other principles often mentioned include: simplicity (keep security mechanisms simple), fail-safe defaults (systems should default to a secure state), open design (security mechanisms should not depend on secrecy of their design), psychological acceptability (security measures should not be overly burdensome to users), etc.)*

**c) Describe how system vulnerabilities can open doors for threat actors in cybersecurity, including the types of threat actors and their impacts. (6 marks)**

**How System Vulnerabilities Open Doors:**
System vulnerabilities are weaknesses or flaws in software, hardware, or configurations that can be exploited by threat actors. When a vulnerability exists:
1.  **Unauthorized Access:** Threat actors can use an exploit (a piece of code or technique) to leverage the vulnerability and gain unauthorized entry into a system or network.
2.  **Privilege Escalation:** Once inside, even with low-level access, attackers can exploit other vulnerabilities to elevate their privileges to administrator or root level, giving them full control.
3.  **Malware Delivery:** Vulnerabilities (e.g., in web browsers or email clients) can be used to silently install malware onto a user's system when they visit a malicious website or open a crafted file.
4.  **Data Exfiltration:** Attackers can exploit vulnerabilities in databases or applications to steal sensitive information.
5.  **Denial of Service:** Some vulnerabilities can be exploited to crash systems or make them unavailable.

**Types of Threat Actors and Their Impacts (related to exploiting vulnerabilities):**

1.  **Cybercriminals:**
    *   **Motivation:** Financial gain.
    *   **Impact:** Exploit vulnerabilities to deploy ransomware (demanding payment), steal financial data (credit card numbers, bank accounts), or sell stolen personal information on the dark web. Can cause significant financial loss and reputational damage to individuals and organizations.

2.  **State-Sponsored Actors / Nation-States:**
    *   **Motivation:** Espionage, intelligence gathering, disrupting critical infrastructure, cyber warfare.
    *   **Impact:** Exploit vulnerabilities (often zero-day vulnerabilities they discover or purchase) to infiltrate government systems, critical infrastructure (e.g., power grids, financial systems), or defense contractors to steal classified information, intellectual property, or cause widespread disruption. Impacts can be severe, affecting national security.

3.  **Hacktivists:**
    *   **Motivation:** Political, social, or ideological agendas.
    *   **Impact:** Exploit vulnerabilities to deface websites, leak sensitive or embarrassing information from organizations they oppose, or launch DoS attacks to disrupt services and draw attention to their cause. Impact is often reputational damage and service disruption.

4.  **Script Kiddies:**
    *   **Motivation:** Curiosity, notoriety, thrill, or to show off skills (often limited).
    *   **Impact:** Use readily available exploit tools and scripts (often without fully understanding them) to exploit known vulnerabilities. While their intent may not be highly sophisticated, they can still cause significant damage, data breaches, or service disruptions if they successfully exploit a vulnerable system.

**d) Discuss how rising cybercrime costs and new attack methods emphasise the need for advanced security training and strategies. (3 marks)**

*   **Rising Cybercrime Costs:** The increasing financial impact of cybercrime (e.g., costs of data breaches, ransomware payments, recovery efforts, regulatory fines, reputational damage) creates a strong business case for investing in better security. Advanced security training and strategies are essential to reduce these costs by preventing incidents or minimizing their impact. Organizations realize that the cost of proactive security measures is often far less than the cost of a successful attack.
*   **New Attack Methods:** Threat actors are constantly evolving their tactics, techniques, and procedures (TTPs), developing more sophisticated and evasive attack methods (e.g., AI-powered attacks, fileless malware, advanced persistent threats - APTs, complex social engineering). Traditional security measures and basic training may no longer be sufficient. Advanced security training is needed to equip IT staff and employees to recognize, defend against, and respond to these new threats. Advanced strategies (like Zero Trust, threat intelligence integration, proactive threat hunting) are required to build resilient defenses against modern adversaries.

In essence, the escalating financial stakes and the dynamic threat landscape necessitate a move beyond basic security to more sophisticated, adaptive, and intelligence-driven security training and strategies to effectively protect assets.

---
I'll proceed with Question 2 of the May/June 2024 paper next.

---

**Question 2 (May/June 2024)**

**a) Identify and explain TWO (2) distinct implementations that fulfil the complementary requirements essential for the protection of physical assets supporting the storage and processing of information. (6 marks)**

The complementary requirements often refer to **prevention** (stopping unauthorized physical access or damage) and **detection/response** (identifying breaches and reacting to them).

1.  **Implementation 1: Multi-Layered Access Control Systems (Prevention)**
    *   **Identification:** This involves deploying a series of physical barriers and authentication mechanisms that an individual must pass through to reach sensitive physical assets (like servers in a data center).
    *   **Explanation:**
        *   **Perimeter Security:** Fences, gates, security guards at the outermost layer.
        *   **Building Access:** Key card/badge readers, security receptionists for entry into the building.
        *   **Data Center/Server Room Access:** More stringent controls like biometric scanners (fingerprint, iris), PIN pads, or multi-factor authentication at the door to the server room.
        *   **Rack-Level Security:** Locked server racks or cages within the server room.
    *   **How it protects:** This layered approach aims to prevent unauthorized individuals from physically reaching the assets. Each layer provides an additional obstacle, making it progressively harder for an intruder to gain access. It focuses on proactively stopping physical breaches.

2.  **Implementation 2: Comprehensive Surveillance and Intrusion Detection Systems (Detection & Response)**
    *   **Identification:** This involves deploying technology to continuously monitor physical spaces and detect any unauthorized access attempts or environmental threats, coupled with a plan to respond to alerts.
    *   **Explanation:**
        *   **CCTV (Closed-Circuit Television):** Cameras strategically placed to monitor perimeters, entry points, and sensitive areas like server rooms. This provides real-time visibility and records events for later investigation.
        *   **Intrusion Detection Sensors:** Motion sensors, door/window contact sensors, glass break detectors that trigger alarms if unauthorized entry or activity is detected.
        *   **Environmental Sensors:** Smoke detectors, heat sensors, water leakage detectors to identify environmental threats.
        *   **Alarm Systems and Monitoring:** Alarms are routed to security personnel or a monitoring station for immediate investigation and response (e.g., dispatching guards, notifying authorities).
    *   **How it protects:** While access controls aim to prevent, surveillance and intrusion detection aim to detect breaches if prevention fails. This allows for a rapid response to contain the incident, minimize damage or loss, and apprehend intruders. The recorded data also aids in post-incident analysis.

These two implementations are complementary: robust access control reduces the likelihood of a breach, while effective surveillance and detection ensure that if a breach does occur (or is attempted), it is identified and responded to quickly.

**b) Given a scenario where a new data centre is being designed, outline THREE (3) physical security measures that should be incorporated to protect against the following threats:**

**(i) Internal (2 marks)**
*   **Threat Example (Internal):** A disgruntled employee or an authorized but careless employee attempting unauthorized access to sensitive areas or equipment, or causing accidental damage.
*   **Physical Security Measure:** **Granular Access Control with Audit Trails:** Implement role-based access controls using smart cards or biometrics for different zones within the data centre (e.g., network operations area, server aisles, storage rooms). Ensure all access attempts (successful and failed) are logged and regularly audited. This limits what even authorized personnel can access and creates accountability.

**(ii) External (2 marks)**
*   **Threat Example (External):** An unauthorized intruder (e.g., industrial spy, thief, vandal) attempting to break into the data centre from outside the building.
*   **Physical Security Measure:** **Robust Perimeter Security:** This includes high fences or walls, controlled entry gates with security guards, anti-climb measures, and clear zones around the building. The building itself should have reinforced doors, windows (or no windows in critical areas), and strong locks.

**(iii) Human (2 marks)**
*   **Threat Example (Human - could be internal or external, malicious or accidental):** Tailgating (an unauthorized person following an authorized person through a secured door), social engineering to gain access, or an employee making a mistake that compromises physical security.
*   **Physical Security Measure:** **Mantraps and Security Awareness Training:**
    *   **Mantraps:** An interlocking two-door system where only one door can be open at a time, and often requiring individual authentication for each person passing through. This prevents tailgating.
    *   **Security Awareness Training:** Train all personnel on physical security policies, how to spot and report suspicious activity, the importance of not allowing tailgating, and verifying the identity of unknown individuals.

**c) Propose TWO (2) fundamental guidelines that an organisation should adhere to concerning the physical security measures for workstations and laptops. (4 marks)**

1.  **Secure Location and Access Control for Workstations; Cable Locks for Laptops:**
    *   **Guideline:** Workstations containing or accessing sensitive information should be located in secure, controlled-access areas. Laptops, being portable, should be secured with cable locks when left unattended in the office (e.g., locking them to a desk) and should never be left visible in unattended vehicles. Implement clean desk policies, requiring employees to lock their screens when away and store sensitive documents securely.
2.  **Encryption and Tracking for Laptops; Strong Authentication for All:**
    *   **Guideline:** All laptops should have full-disk encryption enabled to protect data if the device is lost or stolen. Consider asset tracking software for laptops. Both workstations and laptops must require strong authentication (e.g., complex passwords, multi-factor authentication if possible) to log in.

**d) Identify THREE (3) significant network attacks that potentially impacts on network security. (3 marks)**

1.  **Denial-of-Service (DoS) / Distributed Denial-of-Service (DDoS) Attacks:** Overwhelm network resources, making services unavailable.
2.  **Man-in-the-Middle (MitM) Attacks:** Intercept and potentially alter communication between two parties on the network.
3.  **SQL Injection Attacks:** Exploit vulnerabilities in web applications to manipulate backend databases via network requests, potentially leading to data breaches or unauthorized modifications.

**e) Evaluate the roles of firewalls and Intrusion Detection Systems (IDS) in enhancing network security. Include ONE (1) example of how these components can be effectively implemented. (6 marks)**

**Roles of Firewalls:**
*   **Access Control:** Firewalls act as a primary barrier between a trusted internal network and untrusted external networks (like the internet), or between different segments within an internal network. They inspect network traffic and decide whether to allow or block it based on a pre-defined set of security rules (policies).
*   **Traffic Filtering:** They filter traffic based on criteria such as source/destination IP addresses, port numbers, and protocols. Stateful firewalls also track the state of active connections.
*   **Preventing Unauthorized Access:** By blocking unwanted inbound traffic, firewalls help prevent external attackers from directly accessing internal systems and resources.
*   **Logging and Monitoring:** Firewalls log allowed and denied traffic, which can be used for security monitoring, auditing, and incident investigation.

**Roles of Intrusion Detection Systems (IDS):**
*   **Threat Detection:** IDS monitor network traffic (Network IDS - NIDS) or system activity (Host IDS - HIDS) for suspicious patterns, malicious activities, or policy violations that may indicate an attempted or successful intrusion.
*   **Alerting:** When a potential intrusion is detected, an IDS generates alerts to notify security administrators, enabling them to investigate and respond.
*   **Identifying Known Attacks:** Signature-based IDS can identify known attack patterns and malware.
*   **Identifying Anomalous Behavior:** Anomaly-based IDS can detect deviations from normal baseline behavior, potentially identifying novel or zero-day attacks.
*   **Forensic Analysis:** IDS logs provide valuable information for analyzing security incidents and understanding attack methodologies.

**Evaluation of their Enhancement to Network Security:**
Firewalls and IDS are complementary and crucial for enhancing network security:
*   **Firewalls** are primarily a *preventive* control, acting as a gatekeeper to block known bad traffic and enforce access policies. They are the first line of defense.
*   **IDS** are primarily a *detective* control, acting as a surveillance system to identify malicious activity that might bypass the firewall or originate from within the network. They provide visibility into what's happening on the network.

Together, they create a more robust security posture. The firewall reduces the attack surface by blocking much unwanted traffic, while the IDS monitors the allowed traffic for more subtle or sophisticated threats.

**ONE (1) Example of Effective Implementation:**
*   **Scenario:** A company wants to protect its internal web server from external attacks and monitor for any suspicious activity.
*   **Effective Implementation:**
    1.  **Firewall:** A Next-Generation Firewall (NGFW) is placed at the network perimeter. It is configured with rules to only allow inbound traffic on port 443 (HTTPS) to the specific IP address of the web server. All other inbound ports and traffic to other internal servers are blocked from the internet. The firewall also performs stateful inspection.
    2.  **IDS:** A Network IDS (NIDS) sensor is placed *inside* the firewall, monitoring the traffic that the firewall has allowed to pass through to the internal network, specifically the traffic destined for the web server. The NIDS is configured with signatures for common web attacks (e.g., SQL injection, cross-site scripting) and anomaly detection rules to look for unusual traffic patterns to or from the web server.
    *   **How it works effectively:** The firewall filters out the vast majority of unwanted internet noise and blatant attacks. The NIDS then analyzes the "legitimate" HTTPS traffic allowed by the firewall, looking for more sophisticated attacks embedded within that traffic or signs of compromise. If the NIDS detects a potential SQL injection attempt against the web server, it alerts the security team, who can then investigate and take action, even though the firewall initially allowed the HTTPS connection.

---
I'll proceed with Question 3 of the May/June 2024 paper next.

---

**Question 3 (May/June 2024)**

**a) Apply your understanding of block ciphers and stream ciphers by providing a suitable implementation scenario and example for each. (6 marks)**

1.  **Block Cipher:**
    *   **Suitable Implementation Scenario:** Encrypting a file for secure storage or encrypting a database. In these scenarios, the data is static or processed in chunks, and the fixed block size is manageable. The robustness and well-analyzed security of block ciphers like AES are desirable for protecting valuable stored data.
    *   **Example:**
        *   **Scenario:** A user wants to encrypt a sensitive document (e.g., `confidential_report.docx`) on their hard drive before archiving it.
        *   **Implementation:** The user employs a file encryption utility that uses the **AES (Advanced Encryption Standard)** block cipher in a mode like CBC (Cipher Block Chaining) or GCM (Galois/Counter Mode). The utility divides the document into 128-bit blocks (AES block size). Each block is encrypted using AES with a user-provided key. If the last block is smaller than 128 bits, padding is applied. The result is an encrypted version of the document that can only be decrypted with the correct key.

2.  **Stream Cipher:**
    *   **Suitable Implementation Scenario:** Securing real-time communication streams where data arrives continuously and low latency is important, such as encrypted voice calls (VoIP) or secure video conferencing. Stream ciphers are generally faster and don't require padding, making them efficient for such data streams.
    *   **Example:**
        *   **Scenario:** Two users are having a secure voice conversation over the internet using a VoIP application.
        *   **Implementation:** The VoIP application uses a stream cipher like **ChaCha20** (often part of protocols like TLS 1.3 or for VPNs like WireGuard) to encrypt the voice data. As each user speaks, their voice is digitized into a stream of bits or bytes. The ChaCha20 algorithm generates a pseudorandom keystream using a shared secret key. This keystream is XORed with the voice data stream bit by bit (or byte by byte) to produce the encrypted ciphertext stream, which is then transmitted. The receiving end uses the same keystream (generated from the same shared key) to XOR with the ciphertext and recover the original voice data stream in real-time.

**b) Explain hash algorithms. Determine how a hash algorithm contributes to data integrity with a real-world example. (5 marks)**

**Explanation of Hash Algorithms:**
A hash algorithm (or hash function) is a mathematical function that takes an input (or 'message') of any arbitrary size and produces a fixed-size string of characters, which is typically a sequence of letters and numbers (e.g., a hexadecimal string). This output is called a hash value, hash code, hash sum, or simply a "hash" or "digest."

Key properties of cryptographic hash algorithms include:
1.  **Deterministic:** The same input will always produce the same hash output.
2.  **Fixed-Size Output:** Regardless of the input size, the output hash is always the same length (e.g., SHA-256 always produces a 256-bit hash).
3.  **Efficiency:** Computing the hash for any given input should be fast.
4.  **Pre-image Resistance (One-way):** It should be computationally infeasible to find the original input message given only its hash value.
5.  **Second Pre-image Resistance (Weak Collision Resistance):** Given an input and its hash, it should be computationally infeasible to find a *different* input that produces the same hash.
6.  **Collision Resistance (Strong Collision Resistance):** It should be computationally infeasible to find *any two different* inputs that produce the same hash value.

**How a Hash Algorithm Contributes to Data Integrity:**
Hash algorithms contribute to data integrity by providing a way to detect if data has been altered or tampered with, either accidentally or maliciously.

The process is as follows:
1.  A hash value is calculated for the original data using a specific hash algorithm (e.g., SHA-256).
2.  This original hash value is stored securely or transmitted alongside the data.
3.  Later, to verify the integrity of the data, a new hash value is calculated for the current version of the data using the *same* hash algorithm.
4.  The newly calculated hash is then compared to the original stored/transmitted hash.
    *   If the two hash values are identical, it provides high confidence that the data has not changed since the original hash was computed, thus its integrity is intact.
    *   If the hash values differ, it indicates that the data has been modified in some way. Even a tiny change in the input data (like a single bit) will result in a drastically different hash output (this is known as the avalanche effect).

**Real-World Example:**
**Software Downloads:** When you download a software application or an operating system ISO image from a website, the provider often publishes the MD5, SHA-1, or SHA-256 hash value of the legitimate file alongside the download link.
*   **Process:**
    1.  After downloading the software file (e.g., `installer.exe`), the user can use a utility on their computer to calculate the hash (e.g., SHA-256) of the downloaded file.
    2.  The user then compares the hash value they calculated with the hash value published on the website.
*   **Integrity Check:**
    *   If the hashes match, the user can be reasonably sure that the downloaded file is identical to the original file offered by the provider and has not been corrupted during download or tampered with (e.g., by an attacker injecting malware into it).
    *   If the hashes do not match, it's a strong indication that the file is compromised or corrupted, and the user should not install or run it.

**c) Investigate the roles of a Digital Signature as an electronic validation of the sender's identity. (6 marks)**

A digital signature is a cryptographic mechanism used to verify the authenticity, integrity, and non-repudiation of digital messages or documents. Its role as an electronic validation of the sender's identity (authenticity) is crucial.

The roles are:
1.  **Authentication of Origin (Sender's Identity):**
    *   A digital signature ensures that the message or document indeed originated from the claimed sender. It works using asymmetric cryptography. The sender signs the data (or a hash of the data) using their **private key**. Since the private key is supposed to be known only to the sender, only they could have created that specific signature.
    *   The recipient (or anyone with access to the sender's corresponding **public key**) can verify the signature. If the verification is successful using the sender's public key, it confirms that the signature was created by the holder of the associated private key, thus authenticating the sender.

2.  **Data Integrity:**
    *   While authenticating the sender, the digital signature process also inherently verifies that the message content has not been altered since it was signed. The signature is typically created on a hash of the message. If the message changes, the hash changes, and the signature will no longer validate. This assures the recipient that the message they received is the same one the authenticated sender signed.

3.  **Non-Repudiation:**
    *   This is a strong form of authentication that prevents the sender from later denying that they sent the message or signed the document. Because the signature can only be created with the sender's unique private key, successful verification provides compelling proof that the sender was indeed the originator. The sender cannot plausibly claim that someone else (who doesn't have their private key) created the signature.

**How it works in brief:**
*   **Signing (by Sender):**
    1.  Create a hash of the message.
    2.  Encrypt the hash with the sender's private key. This encrypted hash is the digital signature.
*   **Verification (by Receiver):**
    1.  Decrypt the signature using the sender's public key to get the original hash (Hash1).
    2.  Independently calculate a new hash of the received message (Hash2).
    3.  If Hash1 equals Hash2, the signature is valid, authenticating the sender and confirming integrity.

Thus, a digital signature acts as an electronic, unforgeable "fingerprint" tied to the sender's private key, validating their identity as the source of the signed data.

**d) Simulate the process by which Bob (sender) sends a digitally signed message to Alice (receiver), detailing each step involved from the creation of the message to its receipt and verification by Alice. (8 marks)**

**Scenario:** Bob wants to send a digitally signed email message "Hello Alice, meet me at 3 PM." to Alice.

**Prerequisites:**
*   Bob has a key pair: Bob's Private Key (secret) and Bob's Public Key (known to Alice or publicly available).
*   Alice has a key pair: Alice's Private Key (secret) and Alice's Public Key. (Alice's keys are not directly used in Bob *signing* the message, but Alice would need Bob's public key for verification).
*   Both Bob and Alice have agreed on a hash algorithm (e.g., SHA-256).

**Steps:**

**Bob's Side (Sender):**
1.  **Step 1: Message Creation:** Bob composes his message: "Hello Alice, meet me at 3 PM."
    *   `Original Message = "Hello Alice, meet me at 3 PM."`

2.  **Step 2: Hashing the Message:** Bob's email client/software calculates a hash digest of the original message using the agreed-upon hash algorithm (e.g., SHA-256).
    *   `Message_Digest = HASH("Hello Alice, meet me at 3 PM.")`
    *   *(Let's say this produces a hash value: `0xABC123...`)*

3.  **Step 3: Encrypting the Hash (Creating the Digital Signature):** Bob's software uses **Bob's Private Key** to encrypt the `Message_Digest`. The result is the digital signature.
    *   `Digital_Signature = ENCRYPT(Message_Digest, Bob's_Private_Key)`
    *   *(Let's say this encrypted hash is: `0xDEF456...`)*

4.  **Step 4: Attaching the Signature and Sending:** Bob's email client attaches the `Digital_Signature` to the `Original_Message`. It might also include information about the hash algorithm used and Bob's certificate (which contains Bob's Public Key). The combined package (Original Message + Digital Signature [+ Bob's Certificate/Public Key info]) is then sent to Alice.
    *   `Transmitted_Package = [Original_Message, Digital_Signature, Bob's_Public_Key_Info]`

---
**Transmission:** The email travels from Bob to Alice over the internet.
---

**Alice's Side (Receiver):**
5.  **Step 5: Receiving the Message:** Alice receives the email package from Bob, which contains the `Original_Message` and Bob's `Digital_Signature` (and possibly Bob's public key or a way to get it).
    *   `Received_Message = "Hello Alice, meet me at 3 PM."`
    *   `Received_Signature = 0xDEF456...`

6.  **Step 6: Decrypting the Signature (Retrieving Bob's Original Hash):** Alice's email client/software uses **Bob's Public Key** (which she either has, gets from the message, or retrieves from a trusted directory) to decrypt the `Received_Signature`. This should yield the hash value that Bob originally calculated.
    *   `Decrypted_Hash_from_Signature (Hash1) = DECRYPT(Received_Signature, Bob's_Public_Key)`
    *   *(If successful, this should be `0xABC123...`)*

7.  **Step 7: Calculating Hash of Received Message:** Alice's software independently calculates a new hash digest of the `Received_Message` using the *same hash algorithm* Bob used (e.g., SHA-256).
    *   `Calculated_Hash_of_Received_Message (Hash2) = HASH("Hello Alice, meet me at 3 PM.")`
    *   *(This should also produce `0xABC123...` if the message is unaltered)*

8.  **Step 8: Comparing the Hashes (Verification):** Alice's software compares `Hash1` (obtained from decrypting Bob's signature) with `Hash2` (calculated from the message Alice received).
    *   **If `Hash1 == Hash2`:**
        *   **Authenticity Verified:** The signature is valid. This confirms the message genuinely came from Bob (as only Bob's private key could create a signature that Bob's public key can decrypt to match the message hash).
        *   **Integrity Verified:** The message has not been altered since Bob signed it.
        *   **Non-Repudiation:** Bob cannot easily deny sending this signed message.
        *   Alice can trust the message content and its origin.
    *   **If `Hash1 != Hash2`:**
        *   The signature is invalid. This means either the message was tampered with after Bob signed it, or the signature was not created by Bob (or Bob's public key is incorrect). Alice should not trust the message.

This completes the simulation.

---
I'll proceed with Question 4 of the May/June 2024 paper next.

---

**Question 4 (May/June 2024)**

**a) Explain the concept of password authentication and its vulnerabilities. Discuss TWO (2) common types of attacks on passwords and suggest a countermeasure for each. (7 marks)**

**Concept of Password Authentication:**
Password authentication is a security process used to verify a user's identity. The user claims an identity (e.g., by providing a username) and then provides a secret piece of information (the password) that is theoretically known only to them and the system. The system compares the provided password (often a hash of it) against a stored version (usually a stored hash). If they match, the user is authenticated and granted access. It's based on the principle of "something you know."

**Vulnerabilities of Password Authentication:**
*   **Weak Passwords:** Users often choose simple, easily guessable, or common passwords.
*   **Password Reuse:** Users use the same password across multiple accounts.
*   **Compromise of Stored Passwords:** If the system storing passwords (even as hashes) is breached, attackers can try to crack these hashes offline.
*   **Susceptibility to Social Engineering:** Users can be tricked into revealing their passwords (phishing).
*   **Keystroke Logging:** Malware can capture passwords as they are typed.
*   **Shoulder Surfing:** Passwords can be observed by someone looking over the user's shoulder.
*   **No Proof of User Presence:** A stolen password can be used by anyone, anywhere, without proving the legitimate user is present.

**TWO (2) Common Types of Attacks on Passwords and Countermeasures:**

1.  **Attack Type: Brute-Force Attack**
    *   **Discussion:** Attackers use automated software to systematically try all possible combinations of characters until the correct password is found. This is particularly effective against short or simple passwords.
    *   **Countermeasure: Account Lockout Policy**
        *   **Suggestion:** Implement a policy that temporarily locks an account (e.g., for 15-30 minutes) after a specified number of consecutive failed login attempts (e.g., 3-5 attempts). This significantly slows down brute-force attacks, making them impractical.

2.  **Attack Type: Phishing Attack**
    *   **Discussion:** Attackers send deceptive emails or messages, or create fake websites, to trick users into voluntarily revealing their passwords. The user believes they are interacting with a legitimate service.
    *   **Countermeasure: Multi-Factor Authentication (MFA)**
        *   **Suggestion:** Implement MFA, which requires users to provide two or more verification factors to gain access. For example, after entering their password (something they know), they might also need to provide a one-time code from an authenticator app (something they have) or a biometric scan (something they are). Even if a password is stolen via phishing, the attacker still needs the additional factor(s), which they are unlikely to have.

**b) Critically evaluate the concept of Role-Based Access Control (RBAC) and outline its advantages over Discretionary Access Control (DAC) in the management of user permissions and access privileges. (5 marks)**

**Concept of Role-Based Access Control (RBAC):**
RBAC is an access control model where permissions are assigned to roles, and users are then assigned to these roles. Access to resources is granted based on the roles a user holds, not directly to individual users. Roles are typically defined based on job functions or responsibilities within an organization (e.g., "Accountant," "System Administrator," "Sales Representative").

**Critical Evaluation of RBAC:**
*   **Strengths:** Simplifies permission management in larger organizations by grouping users with similar access needs. Centralizes administration of permissions. Supports the principle of least privilege if roles are well-defined. Easier to audit user access rights.
*   **Weaknesses:** Can become complex if too many roles are created ("role explosion"). May not be granular enough for highly specific access needs that don't neatly fit into a job function. Static nature can be a limitation in dynamic environments where access needs change frequently based on context. Initial setup and role definition can be time-consuming.

**Advantages of RBAC over Discretionary Access Control (DAC):**

1.  **Centralized Administration and Consistency:**
    *   **RBAC:** Permissions are managed centrally by administrators who define roles and assign permissions to those roles. This ensures consistency in how access rights are granted across the organization. Adding or removing a user's access involves simply assigning or revoking roles.
    *   **DAC:** In DAC, the owner of a resource (e.g., the creator of a file) has discretion over who can access it and what permissions they have. This can lead to inconsistent application of access policies, difficulty in auditing, and potential for users to grant inappropriate access. RBAC overcomes this by centralizing control.

2.  **Support for Principle of Least Privilege and Separation of Duties:**
    *   **RBAC:** Roles can be designed to grant only the minimum permissions necessary for a specific job function, inherently supporting the principle of least privilege. RBAC can also facilitate separation of duties by defining roles suchthat no single role has conflicting or overly broad permissions.
    *   **DAC:** While least privilege can be implemented in DAC, it relies on individual resource owners correctly setting permissions, which is prone to error or oversight. DAC doesn't inherently make separation of duties easy to enforce system-wide.

3.  **Simplified User Access Reviews and Auditing:**
    *   **RBAC:** It's easier to review and audit user access. Instead of checking permissions for each user on each resource, auditors can review the permissions assigned to roles and the users assigned to those roles. This simplifies compliance reporting and identifying potential access risks.
    *   **DAC:** Auditing in DAC can be very complex as permissions are scattered and managed by individual resource owners. It's difficult to get a clear, organization-wide view of who has access to what.

4.  **Reduced Administrative Overhead (especially in large organizations):**
    *   **RBAC:** When an employee changes job functions, administrators only need to change their role assignments rather than reconfiguring individual permissions on numerous resources. This is much more efficient than in DAC, where permissions might need to be individually revoked and granted by multiple resource owners.

**c) Examine how Attribute-Based Access Control (ABAC) can address limitations found in Role-Based Access Control (RBAC) for dynamic and complex environments. Provide an example scenario where ABAC would be preferable. (5 marks)**

**How ABAC Addresses RBAC Limitations:**
RBAC, while effective for many scenarios, can be limited in dynamic and complex environments due to its static nature and reliance on predefined roles. ABAC (also known as Policy-Based Access Control - PBAC) addresses these limitations by making access decisions based on policies that combine attributes of the user, the resource, the action, and the environment.

1.  **Fine-Grained Control:** RBAC permissions are tied to roles. ABAC allows for much more granular control because policies can consider a wide range of specific attributes. This overcomes the "one-size-fits-all" nature of roles when very specific access conditions are needed.
2.  **Dynamic and Context-Aware Access:** RBAC roles are usually static. ABAC can make dynamic access decisions based on real-time contextual information (e.g., user's location, time of day, device security posture, current threat level). This is difficult to achieve with RBAC alone.
3.  **Reduced Role Explosion:** In complex environments, trying to capture all necessary permission combinations with RBAC can lead to a proliferation of roles, making management cumbersome. ABAC can handle these complex scenarios with fewer, more expressive policies based on attributes.
4.  **Scalability for Complex Relationships:** ABAC is better suited for scenarios with many users, many resources, and complex relationships between them, where defining roles for every possible interaction becomes impractical.

**Example Scenario Where ABAC Would Be Preferable:**
**Scenario:** A multinational healthcare research organization has a cloud-based data repository containing sensitive patient research data from various global studies. Access needs to be tightly controlled based on multiple factors.

*   **User Attributes:** Researcher's role (e.g., "Lead Investigator," "Data Analyst"), project affiliation, nationality, security clearance level, completion of specific ethics training.
*   **Resource Attributes:** Data sensitivity level (e.g., "Anonymized," "Pseudonymized," "Identifiable"), project ID the data belongs to, data's country of origin (for data sovereignty).
*   **Environmental Attributes:** User's current geographic location (IP address), time of day, security status of the device being used for access.
*   **Action Attributes:** View, download, modify, share.

**Why ABAC is Preferable here:**
*   **RBAC Limitations:** Using RBAC alone would be very difficult. You might need roles like "US_Lead_Investigator_For_ProjectX_With_Ethics_Training_Accessing_Anonymized_Data_From_US_During_WorkHours." This leads to an unmanageable number of roles.
*   **ABAC Solution:** An ABAC policy could state:
    `ALLOW access IF`
    `user.role == "Lead Investigator" AND`
    `user.project_affiliation == resource.project_ID AND`
    `user.ethics_training_completed == true AND`
    `resource.sensitivity_level == "Anonymized" AND`
    `environment.user_location == "USA" AND`
    `environment.time_of_day BETWEEN "0900-1700"`
    `FOR action == "View"`

    This ABAC policy is much more expressive and can dynamically adapt. If a researcher moves to a different project, their `project_affiliation` attribute changes, and the policy automatically adjusts their access without needing a role change. If data sovereignty rules dictate that data from a certain country can only be accessed from within that country, an environmental attribute can enforce this.

**d) Identify the role of security policies in enhancing an organisation's defence against cyber threats. Relate TWO (2) examples of specific policies and their potential impact on organisational security. (8 marks)**

**Role of Security Policies:**
Security policies are high-level statements, rules, and guidelines, approved by senior management, that define an organization's approach to information security. Their role in enhancing defense against cyber threats is foundational:
1.  **Define Acceptable and Unacceptable Behavior:** They clearly outline what users are permitted and prohibited from doing with company IT assets, reducing risky behavior.
2.  **Establish Security Standards and Baselines:** They set the expected level of security for systems, networks, and data handling processes.
3.  **Assign Roles and Responsibilities:** They clarify who is responsible for implementing, maintaining, and complying with security measures.
4.  **Provide a Framework for Controls:** They mandate the implementation of specific security controls (technical, operational, managerial) to mitigate identified risks.
5.  **Support Compliance:** They help the organization meet legal, regulatory, and contractual security obligations.
6.  **Enable Consistent Security Practices:** They ensure a uniform approach to security across the organization.
7.  **Form the Basis for Security Awareness Training:** Policies are the foundation for educating users about their security obligations.

**TWO (2) Examples of Specific Policies and Their Potential Impact:**

1.  **Policy Example: Password Management Policy**
    *   **Content:** This policy would specify requirements for password creation (e.g., minimum length, complexity including uppercase, lowercase, numbers, symbols), password change frequency, prohibition of password sharing, rules against writing passwords down, and procedures for password resets. It might also mandate the use of password managers or multi-factor authentication (MFA).
    *   **Potential Impact on Organisational Security:**
        *   **Positive Impact:**
            *   **Stronger Credentials:** Enforces the use of strong, unique passwords, making brute-force and dictionary attacks much harder for attackers.
            *   **Reduced Credential Stuffing Risk:** If combined with a directive against password reuse and support for MFA, it significantly reduces the risk of compromised credentials from one breach being used to access other company systems.
            *   **Improved Accountability:** Clear rules reduce ambiguity about password security responsibilities.
        *   **Potential Negative Impact (if poorly implemented/enforced):** If too complex or frequently changed without good tools (like password managers), users might resort to insecure practices like writing passwords down or using predictable patterns, inadvertently weakening security.

2.  **Policy Example: Acceptable Use Policy (AUP)**
    *   **Content:** This policy defines how employees are permitted to use company IT resources, including computers, networks, internet access, email, and software. It typically prohibits activities like installing unauthorized software, accessing malicious or inappropriate websites, sharing confidential company information inappropriately, or using company resources for extensive personal use or illegal activities.
    *   **Potential Impact on Organisational Security:**
        *   **Positive Impact:**
            *   **Reduced Malware Infections:** By prohibiting the download/installation of unauthorized software and access to risky websites, it reduces the avenues for malware entry.
            *   **Minimized Data Leakage:** Rules against inappropriate sharing of information help protect sensitive company data.
            *   **Legal Protection:** Provides a basis for disciplinary action if employees violate the policy, and can offer some legal protection for the company.
            *   **Network Performance:** Limiting non-business use can help preserve network bandwidth for legitimate business purposes.
        *   **Potential Negative Impact (if overly restrictive without clear business justification):** Could hinder productivity if it unnecessarily blocks legitimate tools or websites needed for work. Requires clear communication and consistent enforcement to be effective.

---
This completes the May/June 2024 paper. We have now gone through all the provided exam papers. Let me know if you have any other questions!
